import logging

from mem0.memory.utils import format_entities, sanitize_relationship_for_cypher

# Required keys for a valid entity from LLM tool calls
_ENTITY_REQUIRED_KEYS = {"source", "relationship", "destination"}

try:
    from falkordb import FalkorDB
except ImportError:
    raise ImportError("falkordb is not installed. Please install it using pip install falkordb")

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("rank_bm25 is not installed. Please install it using pip install rank-bm25")

from mem0.graphs.tools import (
    DELETE_MEMORY_STRUCT_TOOL_GRAPH,
    DELETE_MEMORY_TOOL_GRAPH,
    EXTRACT_ENTITIES_STRUCT_TOOL,
    EXTRACT_ENTITIES_TOOL,
    RELATIONS_STRUCT_TOOL,
    RELATIONS_TOOL,
)
from mem0.graphs.utils import EXTRACT_RELATIONS_PROMPT, get_delete_messages
from mem0.utils.factory import EmbedderFactory, LlmFactory

logger = logging.getLogger(__name__)


class MemoryGraph:
    def __init__(self, config):
        self.config = config
        graph_config = self.config.graph_store.config

        # Connect to FalkorDB
        self.driver = FalkorDB(
            host=graph_config.host,
            port=graph_config.port,
            username=graph_config.username,
            password=graph_config.password,
        )
        self.graph = self.driver.select_graph(graph_config.graph_name)

        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider, self.config.embedder.config, self.config.vector_store.config
        )

        # Create vector index on the embedding property for __Entity__ nodes
        try:
            embedding_dim = self.config.embedder.config.get("embedding_dims", 1536)
            if embedding_dim == 1536 and self.config.embedder.provider != "openai":
                logger.warning(
                    "embedding_dims not set, defaulting to 1536 (OpenAI ada-002). "
                    "If using a different embedder, set embedding_dims explicitly to avoid dimension mismatch."
                )
            self.graph.create_node_vector_index("__Entity__", "embedding", dim=embedding_dim, similarity_function="cosine")
        except Exception:
            pass  # Index may already exist

        # Create range indexes for filter properties
        for prop in ("user_id", "name"):
            try:
                self.graph.create_node_range_index("__Entity__", prop)
            except Exception:
                pass  # Index may already exist

        # Default to openai if no specific provider is configured
        self.llm_provider = "openai"
        if self.config.llm and self.config.llm.provider:
            self.llm_provider = self.config.llm.provider
        if self.config.graph_store and self.config.graph_store.llm and self.config.graph_store.llm.provider:
            self.llm_provider = self.config.graph_store.llm.provider

        # Get LLM config with proper null checks
        llm_config = None
        if self.config.graph_store and self.config.graph_store.llm and hasattr(self.config.graph_store.llm, "config"):
            llm_config = self.config.graph_store.llm.config
        elif hasattr(self.config.llm, "config"):
            llm_config = self.config.llm.config
        self.llm = LlmFactory.create(self.llm_provider, llm_config)
        self.user_id = None
        # Use threshold from graph_store config, default to 0.7 for backward compatibility
        self.threshold = self.config.graph_store.threshold if hasattr(self.config.graph_store, 'threshold') else 0.7

    def add(self, data, filters):
        """
        Adds data to the graph.

        Args:
            data (str): The data to add to the graph.
            filters (dict): A dictionary containing filters to be applied during the addition.
        """
        entity_type_map = self._retrieve_nodes_from_data(data, filters)
        to_be_added = self._establish_nodes_relations_from_data(data, filters, entity_type_map)
        search_output = self._search_graph_db(node_list=list(entity_type_map.keys()), filters=filters)
        to_be_deleted = self._get_delete_entities_from_search_output(search_output, data, filters)

        deleted_entities = self._delete_entities(to_be_deleted, filters)
        added_entities = self._add_entities(to_be_added, filters, entity_type_map)

        return {"deleted_entities": deleted_entities, "added_entities": added_entities}

    def search(self, query, filters, limit=100):
        """
        Search for memories and related graph data.

        Args:
            query (str): Query to search for.
            filters (dict): A dictionary containing filters to be applied during the search.
            limit (int): The maximum number of nodes and relationships to retrieve. Defaults to 100.

        Returns:
            dict: A dictionary containing:
                - "contexts": List of search results from the base data store.
                - "entities": List of related graph data based on the query.
        """
        entity_type_map = self._retrieve_nodes_from_data(query, filters)
        search_output = self._search_graph_db(node_list=list(entity_type_map.keys()), filters=filters)

        if not search_output:
            return []

        search_outputs_sequence = [
            [item["source"], item["relationship"], item["destination"]] for item in search_output
        ]
        bm25 = BM25Okapi(search_outputs_sequence)

        tokenized_query = query.split(" ")
        reranked_results = bm25.get_top_n(tokenized_query, search_outputs_sequence, n=5)

        search_results = []
        for item in reranked_results:
            search_results.append({"source": item[0], "relationship": item[1], "destination": item[2]})

        logger.info(f"Returned {len(search_results)} search results")

        return search_results

    def delete(self, data, filters):
        """
        Delete graph entities associated with the given memory text.

        Extracts entities and relationships from the memory text using the same
        pipeline as add(), then soft-deletes the matching relationships in the graph.

        Args:
            data (str): The memory text whose graph entities should be removed.
            filters (dict): Scope filters (user_id, agent_id, run_id).
        """
        try:
            entity_type_map = self._retrieve_nodes_from_data(data, filters)
            if not entity_type_map:
                logger.debug("No entities found in memory text, skipping graph cleanup")
                return
            to_be_deleted = self._establish_nodes_relations_from_data(data, filters, entity_type_map)
            if to_be_deleted:
                self._delete_entities(to_be_deleted, filters)
        except Exception as e:
            logger.error(f"Error during graph cleanup for memory delete: {e}")

    def delete_all(self, filters):
        """Delete all nodes and relationships for a user (hard delete).

        Unlike delete() which soft-deletes individual relationships,
        delete_all() permanently removes all data including soft-deleted
        history. This is intentional — consistent with upstream behavior.
        """
        # Build WHERE conditions for filtering
        where_conditions = ["n.user_id = $user_id"]
        params = {"user_id": filters["user_id"]}
        if filters.get("agent_id"):
            where_conditions.append("n.agent_id = $agent_id")
            params["agent_id"] = filters["agent_id"]
        if filters.get("run_id"):
            where_conditions.append("n.run_id = $run_id")
            params["run_id"] = filters["run_id"]
        where_clause = " AND ".join(where_conditions)

        cypher = f"""
        MATCH (n:__Entity__)
        WHERE {where_clause}
        DETACH DELETE n
        """
        self.graph.query(cypher, params)

    def get_all(self, filters, limit=100):
        """
        Retrieves all nodes and relationships from the graph database based on optional filtering criteria.

        Args:
            filters (dict): A dictionary containing filters to be applied during the retrieval.
            limit (int): The maximum number of nodes and relationships to retrieve. Defaults to 100.

        Returns:
            list: A list of dictionaries, each containing source, relationship, and target.
        """
        where_conditions = ["n.user_id = $user_id", "m.user_id = $user_id"]
        params = {"user_id": filters["user_id"], "limit": limit}
        if filters.get("agent_id"):
            where_conditions.append("n.agent_id = $agent_id")
            where_conditions.append("m.agent_id = $agent_id")
            params["agent_id"] = filters["agent_id"]
        if filters.get("run_id"):
            where_conditions.append("n.run_id = $run_id")
            where_conditions.append("m.run_id = $run_id")
            params["run_id"] = filters["run_id"]
        where_clause = " AND ".join(where_conditions)

        query = f"""
        MATCH (n:__Entity__)-[r]->(m:__Entity__)
        WHERE {where_clause} AND (r.valid IS NULL OR r.valid = true)
        RETURN n.name AS source, type(r) AS relationship, m.name AS target
        LIMIT $limit
        """
        result = self.graph.query(query, params)

        final_results = []
        for row in result.result_set:
            final_results.append(
                {
                    "source": row[0],
                    "relationship": row[1],
                    "target": row[2],
                }
            )

        logger.info(f"Retrieved {len(final_results)} relationships")

        return final_results

    def _retrieve_nodes_from_data(self, data, filters):
        """Extracts all the entities mentioned in the query."""
        _tools = [EXTRACT_ENTITIES_TOOL]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [EXTRACT_ENTITIES_STRUCT_TOOL]
        search_results = self.llm.generate_response(
            messages=[
                {
                    "role": "system",
                    "content": f"You are a smart assistant who understands entities and their types in a given text. If user message contains self reference such as 'I', 'me', 'my' etc. then use {filters['user_id']} as the source entity. Extract all the entities from the text. ***DO NOT*** answer the question itself if the given text is a question.",
                },
                {"role": "user", "content": data},
            ],
            tools=_tools,
        )

        entity_type_map = {}

        try:
            for tool_call in search_results["tool_calls"]:
                if tool_call["name"] != "extract_entities":
                    continue
                for item in tool_call.get("arguments", {}).get("entities", []):
                    entity_type_map[item["entity"]] = item["entity_type"]
        except Exception as e:
            logger.exception(
                f"Error in search tool: {e}, llm_provider={self.llm_provider}, search_results={search_results}"
            )

        entity_type_map = {k.lower().replace(" ", "_"): v.lower().replace(" ", "_") for k, v in entity_type_map.items()}
        logger.debug(f"Entity type map: {entity_type_map}\n search_results={search_results}")
        return entity_type_map

    def _establish_nodes_relations_from_data(self, data, filters, entity_type_map):
        """Establish relations among the extracted nodes."""

        # Compose user identification string for prompt
        user_identity = f"user_id: {filters['user_id']}"
        if filters.get("agent_id"):
            user_identity += f", agent_id: {filters['agent_id']}"
        if filters.get("run_id"):
            user_identity += f", run_id: {filters['run_id']}"

        if self.config.graph_store.custom_prompt:
            system_content = EXTRACT_RELATIONS_PROMPT.replace("USER_ID", user_identity)
            # Add the custom prompt line if configured
            system_content = system_content.replace("CUSTOM_PROMPT", f"4. {self.config.graph_store.custom_prompt}")
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": data},
            ]
        else:
            system_content = EXTRACT_RELATIONS_PROMPT.replace("USER_ID", user_identity)
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"List of entities: {list(entity_type_map.keys())}. \n\nText: {data}"},
            ]

        _tools = [RELATIONS_TOOL]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [RELATIONS_STRUCT_TOOL]

        extracted_entities = self.llm.generate_response(
            messages=messages,
            tools=_tools,
        )

        entities = []
        if extracted_entities.get("tool_calls"):
            entities = extracted_entities["tool_calls"][0].get("arguments", {}).get("entities", [])

        # Filter out entities with missing or invalid fields (defensive against incomplete LLM tool calls)
        valid_entities = []
        for entity in entities:
            missing = _ENTITY_REQUIRED_KEYS - entity.keys()
            if missing:
                logger.warning("[_establish_nodes_relations] Skipping entity with missing fields: missing=%s, entity=%s", missing, entity)
            elif not all(isinstance(entity.get(k), str) and entity[k].strip() for k in _ENTITY_REQUIRED_KEYS):
                logger.warning("[_establish_nodes_relations] Skipping entity with empty/non-string values: entity=%s", entity)
            else:
                valid_entities.append(entity)
        entities = valid_entities

        entities = self._remove_spaces_from_entities(entities)
        logger.debug(f"Extracted entities: {entities}")
        return entities

    def _search_graph_db(self, node_list, filters, limit=100):
        """Search similar nodes and their respective incoming and outgoing relations."""
        result_relations = []

        # Build WHERE conditions for filtering
        filter_conditions = ["n.user_id = $user_id"]
        if filters.get("agent_id"):
            filter_conditions.append("n.agent_id = $agent_id")
        if filters.get("run_id"):
            filter_conditions.append("n.run_id = $run_id")

        # Build relationship target filter conditions
        target_conditions = ["m.user_id = $user_id"]
        if filters.get("agent_id"):
            target_conditions.append("m.agent_id = $agent_id")
        if filters.get("run_id"):
            target_conditions.append("m.run_id = $run_id")
        target_where = " AND ".join(target_conditions)

        for node in node_list:
            n_embedding = self.embedding_model.embed(node)

            # Use FalkorDB vector search to find similar nodes
            cypher_query = f"""
            CALL db.idx.vector.queryNodes('__Entity__', 'embedding', $topK, vecf32($n_embedding))
            YIELD node AS n, score AS similarity
            WHERE {" AND ".join(filter_conditions)} AND similarity >= $threshold
            CALL {{
                WITH n
                MATCH (n)-[r]->(m:__Entity__)
                WHERE {target_where} AND (r.valid IS NULL OR r.valid = true)
                RETURN n.name AS source, id(n) AS source_id, type(r) AS relationship, id(r) AS relation_id, m.name AS destination, id(m) AS destination_id
                UNION
                WITH n
                MATCH (n)<-[r]-(m:__Entity__)
                WHERE {target_where} AND (r.valid IS NULL OR r.valid = true)
                RETURN m.name AS source, id(m) AS source_id, type(r) AS relationship, id(r) AS relation_id, n.name AS destination, id(n) AS destination_id
            }}
            WITH DISTINCT source, source_id, relationship, relation_id, destination, destination_id, similarity
            RETURN source, source_id, relationship, relation_id, destination, destination_id, similarity
            ORDER BY similarity DESC
            LIMIT $limit
            """

            params = {
                "n_embedding": n_embedding,
                "threshold": self.threshold,
                "user_id": filters["user_id"],
                "limit": limit,
                "topK": limit,
            }
            if filters.get("agent_id"):
                params["agent_id"] = filters["agent_id"]
            if filters.get("run_id"):
                params["run_id"] = filters["run_id"]

            result = self.graph.query(cypher_query, params)
            # Parse FalkorDB result_set into dicts
            if result.result_set:
                header = [col[1] for col in result.header]
                for row in result.result_set:
                    row_dict = dict(zip(header, row))
                    result_relations.append(row_dict)

        return result_relations

    def _get_delete_entities_from_search_output(self, search_output, data, filters):
        """Get the entities to be deleted from the search output."""
        search_output_string = format_entities(search_output)

        # Compose user identification string for prompt
        user_identity = f"user_id: {filters['user_id']}"
        if filters.get("agent_id"):
            user_identity += f", agent_id: {filters['agent_id']}"
        if filters.get("run_id"):
            user_identity += f", run_id: {filters['run_id']}"

        system_prompt, user_prompt = get_delete_messages(search_output_string, data, user_identity)

        _tools = [DELETE_MEMORY_TOOL_GRAPH]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [
                DELETE_MEMORY_STRUCT_TOOL_GRAPH,
            ]

        memory_updates = self.llm.generate_response(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=_tools,
        )

        to_be_deleted = []
        for item in memory_updates.get("tool_calls", []):
            if item.get("name") == "delete_graph_memory":
                to_be_deleted.append(item.get("arguments"))
        # Clean entities formatting
        to_be_deleted = self._remove_spaces_from_entities(to_be_deleted)
        logger.debug(f"Deleted relationships: {to_be_deleted}")
        return to_be_deleted

    def _delete_entities(self, to_be_deleted, filters):
        """Soft-delete relationships by marking r.valid = false."""
        user_id = filters["user_id"]
        agent_id = filters.get("agent_id", None)
        run_id = filters.get("run_id", None)
        results = []

        for item in to_be_deleted:
            # Defensive: skip items with missing or invalid fields
            missing = _ENTITY_REQUIRED_KEYS - item.keys()
            if missing:
                logger.warning("[_delete_entities] Skipping item with missing fields: missing=%s, item=%s", missing, item)
                continue
            if not all(isinstance(item.get(k), str) and item[k].strip() for k in _ENTITY_REQUIRED_KEYS):
                logger.warning("[_delete_entities] Skipping item with empty/non-string values: item=%s", item)
                continue

            source = item["source"]
            destination = item["destination"]
            relationship = item["relationship"]

            # Build WHERE conditions for source and destination
            where_conditions = [
                "n.name = $source_name",
                "n.user_id = $user_id",
                "m.name = $dest_name",
                "m.user_id = $user_id",
                # Only soft-delete relationships that are still valid
                "(r.valid IS NULL OR r.valid = true)",
            ]
            params = {
                "source_name": source,
                "dest_name": destination,
                "user_id": user_id,
            }

            if agent_id:
                where_conditions.append("n.agent_id = $agent_id")
                where_conditions.append("m.agent_id = $agent_id")
                params["agent_id"] = agent_id
            if run_id:
                where_conditions.append("n.run_id = $run_id")
                where_conditions.append("m.run_id = $run_id")
                params["run_id"] = run_id

            where_clause = " AND ".join(where_conditions)

            # Soft-delete: mark relationship as invalid instead of removing it,
            # enabling temporal reasoning over historical graph state.
            cypher = f"""
            MATCH (n:__Entity__)-[r:`{relationship}`]->(m:__Entity__)
            WHERE {where_clause}
            SET r.valid = false, r.invalidated_at = timestamp()
            RETURN
                n.name AS source,
                m.name AS target,
                type(r) AS relationship
            """

            result = self.graph.query(cypher, params)
            results.append(result.result_set)

        return results

    def _add_entities(self, to_be_added, filters, entity_type_map):
        """Add the new entities to the graph. Merge the nodes if they already exist."""
        user_id = filters["user_id"]
        agent_id = filters.get("agent_id", None)
        run_id = filters.get("run_id", None)
        results = []
        for item in to_be_added:
            # Defensive: skip items with missing or invalid fields
            missing = _ENTITY_REQUIRED_KEYS - item.keys()
            if missing:
                logger.warning("[_add_entities] Skipping item with missing fields: missing=%s, item=%s", missing, item)
                continue
            if not all(isinstance(item.get(k), str) and item[k].strip() for k in _ENTITY_REQUIRED_KEYS):
                logger.warning("[_add_entities] Skipping item with empty/non-string values: item=%s", item)
                continue

            # entities
            source = item["source"]
            destination = item["destination"]
            relationship = item["relationship"]

            # types
            source_type = entity_type_map.get(source, "__User__")
            destination_type = entity_type_map.get(destination, "__User__")

            # embeddings
            source_embedding = self.embedding_model.embed(source)
            dest_embedding = self.embedding_model.embed(destination)

            # search for the nodes with the closest embeddings
            source_node_search_result = self._search_source_node(source_embedding, filters, threshold=self.threshold)
            destination_node_search_result = self._search_destination_node(dest_embedding, filters, threshold=self.threshold)

            if not destination_node_search_result and source_node_search_result:
                # Source found, destination not found -- build destination MERGE props dynamically
                dest_merge_props = ["name: $destination_name", "user_id: $user_id"]
                params = {
                    "source_id": source_node_search_result[0]["id(source_candidate)"],
                    "destination_name": destination,
                    "destination_embedding": dest_embedding,
                    "destination_type": destination_type,
                    "user_id": user_id,
                }
                if agent_id:
                    dest_merge_props.append("agent_id: $agent_id")
                    params["agent_id"] = agent_id
                if run_id:
                    dest_merge_props.append("run_id: $run_id")
                    params["run_id"] = run_id
                dest_merge_props_str = ", ".join(dest_merge_props)

                # Use MATCH for existing source, MERGE for new destination
                cypher = f"""
                MATCH (source:__Entity__)
                WHERE id(source) = $source_id
                SET source.mentions = CASE WHEN source.mentions IS NULL THEN 1 ELSE source.mentions + 1 END
                WITH source
                MERGE (destination:__Entity__ {{{dest_merge_props_str}}})
                ON CREATE SET
                    destination.created = timestamp(),
                    destination.mentions = 1,
                    destination.entity_type = $destination_type
                ON MATCH SET
                    destination.mentions = CASE WHEN destination.mentions IS NULL THEN 1 ELSE destination.mentions + 1 END
                WITH source, destination
                CALL db.create.setNodeVectorProperty(destination, 'embedding', vecf32($destination_embedding))
                WITH source, destination
                MERGE (source)-[r:`{relationship}`]->(destination)
                ON CREATE SET
                    r.created_at = timestamp(),
                    r.updated_at = timestamp(),
                    r.mentions = 1,
                    r.valid = true
                ON MATCH SET
                    r.mentions = CASE WHEN r.mentions IS NULL THEN 1 ELSE r.mentions + 1 END,
                    r.valid = true,
                    r.updated_at = timestamp(),
                    r.invalidated_at = null
                RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                """

            elif destination_node_search_result and not source_node_search_result:
                # Destination found, source not found -- build source MERGE props dynamically
                source_merge_props = ["name: $source_name", "user_id: $user_id"]
                params = {
                    "destination_id": destination_node_search_result[0]["id(destination_candidate)"],
                    "source_name": source,
                    "source_embedding": source_embedding,
                    "source_type": source_type,
                    "user_id": user_id,
                }
                if agent_id:
                    source_merge_props.append("agent_id: $agent_id")
                    params["agent_id"] = agent_id
                if run_id:
                    source_merge_props.append("run_id: $run_id")
                    params["run_id"] = run_id
                source_merge_props_str = ", ".join(source_merge_props)

                cypher = f"""
                MATCH (destination:__Entity__)
                WHERE id(destination) = $destination_id
                SET destination.mentions = CASE WHEN destination.mentions IS NULL THEN 1 ELSE destination.mentions + 1 END
                WITH destination
                MERGE (source:__Entity__ {{{source_merge_props_str}}})
                ON CREATE SET
                    source.created = timestamp(),
                    source.mentions = 1,
                    source.entity_type = $source_type
                ON MATCH SET
                    source.mentions = CASE WHEN source.mentions IS NULL THEN 1 ELSE source.mentions + 1 END
                WITH source, destination
                CALL db.create.setNodeVectorProperty(source, 'embedding', vecf32($source_embedding))
                WITH source, destination
                MERGE (source)-[r:`{relationship}`]->(destination)
                ON CREATE SET
                    r.created_at = timestamp(),
                    r.updated_at = timestamp(),
                    r.mentions = 1,
                    r.valid = true
                ON MATCH SET
                    r.mentions = CASE WHEN r.mentions IS NULL THEN 1 ELSE r.mentions + 1 END,
                    r.valid = true,
                    r.updated_at = timestamp(),
                    r.invalidated_at = null
                RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                """

            elif source_node_search_result and destination_node_search_result:
                # Both found -- just create/merge the relationship (no MERGE node needed)
                params = {
                    "source_id": source_node_search_result[0]["id(source_candidate)"],
                    "destination_id": destination_node_search_result[0]["id(destination_candidate)"],
                    "user_id": user_id,
                }
                if agent_id:
                    params["agent_id"] = agent_id
                if run_id:
                    params["run_id"] = run_id

                cypher = f"""
                MATCH (source:__Entity__)
                WHERE id(source) = $source_id
                SET source.mentions = CASE WHEN source.mentions IS NULL THEN 1 ELSE source.mentions + 1 END
                WITH source
                MATCH (destination:__Entity__)
                WHERE id(destination) = $destination_id
                SET destination.mentions = CASE WHEN destination.mentions IS NULL THEN 1 ELSE destination.mentions + 1 END
                MERGE (source)-[r:`{relationship}`]->(destination)
                ON CREATE SET
                    r.created_at = timestamp(),
                    r.updated_at = timestamp(),
                    r.mentions = 1,
                    r.valid = true
                ON MATCH SET
                    r.mentions = CASE WHEN r.mentions IS NULL THEN 1 ELSE r.mentions + 1 END,
                    r.valid = true,
                    r.updated_at = timestamp(),
                    r.invalidated_at = null
                RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                """

            else:
                # Neither found -- build MERGE props for both source and destination
                source_merge_props = ["name: $source_name", "user_id: $user_id"]
                dest_merge_props = ["name: $dest_name", "user_id: $user_id"]
                params = {
                    "source_name": source,
                    "dest_name": destination,
                    "source_embedding": source_embedding,
                    "dest_embedding": dest_embedding,
                    "source_type": source_type,
                    "destination_type": destination_type,
                    "user_id": user_id,
                }
                if agent_id:
                    source_merge_props.append("agent_id: $agent_id")
                    dest_merge_props.append("agent_id: $agent_id")
                    params["agent_id"] = agent_id
                if run_id:
                    source_merge_props.append("run_id: $run_id")
                    dest_merge_props.append("run_id: $run_id")
                    params["run_id"] = run_id
                source_merge_props_str = ", ".join(source_merge_props)
                dest_merge_props_str = ", ".join(dest_merge_props)

                cypher = f"""
                MERGE (source:__Entity__ {{{source_merge_props_str}}})
                ON CREATE SET source.created = timestamp(),
                            source.mentions = 1,
                            source.entity_type = $source_type
                ON MATCH SET source.mentions = CASE WHEN source.mentions IS NULL THEN 1 ELSE source.mentions + 1 END
                WITH source
                CALL db.create.setNodeVectorProperty(source, 'embedding', vecf32($source_embedding))
                WITH source
                MERGE (destination:__Entity__ {{{dest_merge_props_str}}})
                ON CREATE SET destination.created = timestamp(),
                            destination.mentions = 1,
                            destination.entity_type = $destination_type
                ON MATCH SET destination.mentions = CASE WHEN destination.mentions IS NULL THEN 1 ELSE destination.mentions + 1 END
                WITH source, destination
                CALL db.create.setNodeVectorProperty(destination, 'embedding', vecf32($dest_embedding))
                WITH source, destination
                MERGE (source)-[r:`{relationship}`]->(destination)
                ON CREATE SET
                    r.created_at = timestamp(),
                    r.updated_at = timestamp(),
                    r.mentions = 1,
                    r.valid = true
                ON MATCH SET
                    r.mentions = CASE WHEN r.mentions IS NULL THEN 1 ELSE r.mentions + 1 END,
                    r.valid = true,
                    r.updated_at = timestamp(),
                    r.invalidated_at = null
                RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                """


            result = self.graph.query(cypher, params)
            results.append(result.result_set)
        return results

    def _remove_spaces_from_entities(self, entity_list):
        for item in entity_list:
            item["source"] = item["source"].lower().replace(" ", "_")
            # Use the sanitization function for relationships to handle special characters
            item["relationship"] = sanitize_relationship_for_cypher(item["relationship"].lower().replace(" ", "_"))
            item["destination"] = item["destination"].lower().replace(" ", "_")
        return entity_list

    def _search_source_node(self, source_embedding, filters, threshold=0.9):
        # Build WHERE conditions
        filter_conditions = ["source_candidate.user_id = $user_id"]
        if filters.get("agent_id"):
            filter_conditions.append("source_candidate.agent_id = $agent_id")
        if filters.get("run_id"):
            filter_conditions.append("source_candidate.run_id = $run_id")
        filter_where = " AND ".join(filter_conditions)

        cypher = f"""
            CALL db.idx.vector.queryNodes('__Entity__', 'embedding', 1, vecf32($source_embedding))
            YIELD node AS source_candidate, score AS source_similarity
            WHERE {filter_where} AND source_similarity >= $threshold
            RETURN id(source_candidate), source_candidate.name, source_similarity
            """

        params = {
            "source_embedding": source_embedding,
            "user_id": filters["user_id"],
            "threshold": threshold,
        }
        if filters.get("agent_id"):
            params["agent_id"] = filters["agent_id"]
        if filters.get("run_id"):
            params["run_id"] = filters["run_id"]

        result = self.graph.query(cypher, params)
        if result.result_set:
            header = [col[1] for col in result.header]
            return [dict(zip(header, row)) for row in result.result_set]
        return []

    def _search_destination_node(self, destination_embedding, filters, threshold=0.9):
        # Build WHERE conditions
        filter_conditions = ["destination_candidate.user_id = $user_id"]
        if filters.get("agent_id"):
            filter_conditions.append("destination_candidate.agent_id = $agent_id")
        if filters.get("run_id"):
            filter_conditions.append("destination_candidate.run_id = $run_id")
        filter_where = " AND ".join(filter_conditions)

        cypher = f"""
            CALL db.idx.vector.queryNodes('__Entity__', 'embedding', 1, vecf32($destination_embedding))
            YIELD node AS destination_candidate, score AS destination_similarity
            WHERE {filter_where} AND destination_similarity >= $threshold
            RETURN id(destination_candidate), destination_candidate.name, destination_similarity
            """

        params = {
            "destination_embedding": destination_embedding,
            "user_id": filters["user_id"],
            "threshold": threshold,
        }
        if filters.get("agent_id"):
            params["agent_id"] = filters["agent_id"]
        if filters.get("run_id"):
            params["run_id"] = filters["run_id"]

        result = self.graph.query(cypher, params)
        if result.result_set:
            header = [col[1] for col in result.header]
            return [dict(zip(header, row)) for row in result.result_set]
        return []

    # Reset is not defined in base.py
    def reset(self):
        """Reset the graph by clearing all nodes and relationships."""
        logger.warning("Clearing graph...")
        cypher_query = """
        MATCH (n) DETACH DELETE n
        """
        return self.graph.query(cypher_query)
