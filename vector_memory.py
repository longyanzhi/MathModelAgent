#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vector memory system implementation.
"""

import os

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
from datetime import datetime

from utils.logger import logger


class VectorMemory:
    """Vector memory system."""

    DEFAULT_MODEL_NAME = 'all-MiniLM-L6-v2'
    DEFAULT_MAX_MEMORIES = 1000
    DEFAULT_SIMILARITY_THRESHOLD = 0.7
    TIME_DECAY_FACTOR = 0.05
    ACCESS_COUNT_FACTOR = 0.1

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        """Initialize the vector memory system.

        Loading order:
        1. Local cache only (no network call) -- avoids [WinError 10038] on
           Windows when the Werkzeug reloader kills the subprocess mid-request.
        2. Hugging Face / configured mirror.
        3. Degraded mode: semantic memory disabled, meetings continue, but
           context search returns empty.
        """
        self.memory_embeddings: List[np.ndarray] = []
        self.memory_items: List[Dict] = []
        self.memory_index: Dict[str, List[int]] = {}

        self.max_memories = self.DEFAULT_MAX_MEMORIES
        self.similarity_threshold = self.DEFAULT_SIMILARITY_THRESHOLD
        self.model = None
        self._disabled_reason: Optional[str] = None

        # 1) Prefer local cache (no network)
        try:
            self.model = SentenceTransformer(model_name, local_files_only=True)
            logger.info(f"VectorMemory loaded '{model_name}' from local cache.")
            return
        except Exception as cache_err:
            logger.debug(f"VectorMemory: local cache miss for '{model_name}': {cache_err}")

        # 2) Try downloading from Hugging Face / mirror
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"VectorMemory downloaded '{model_name}'.")
        except Exception as net_err:
            self._disabled_reason = f"{type(net_err).__name__}: {net_err}"
            logger.warning(
                f"VectorMemory could not load '{model_name}'; "
                f"semantic memory disabled ({self._disabled_reason}). "
                f"Meetings will continue but context search will be empty."
            )

    @property
    def enabled(self) -> bool:
        """Whether the embedding model is loaded and available."""
        return self.model is not None

    def add_conversation(self, role: str, content: str, turn: int,
                         conversation_type: str = "general"):
        """Add a conversation turn to the vector memory system."""
        if not self.enabled:
            # In degraded mode, keep the most recent raw items so that
            # get_context_for_agent can build a truncated context. Items beyond
            # max_memories are simply dropped.
            memory_item = self._create_memory_item(
                role, content, np.zeros(1, dtype=np.float32),
                turn, conversation_type,
            )
            self.memory_items.append(memory_item)
            self._update_index(memory_item)
            if len(self.memory_items) > self.max_memories:
                self.memory_items.pop(0)
                # IDs follow the shift; simple pop-head, no complex pruning.
                self._rebuild_index()
            return

        embedding = self.model.encode(content)
        memory_item = self._create_memory_item(role, content, embedding, turn, conversation_type)

        if not self._is_duplicate(embedding, content):
            self.memory_embeddings.append(embedding)
            self.memory_items.append(memory_item)
            self._update_index(memory_item)
            if len(self.memory_items) > self.max_memories:
                self._prune_memories()

    def _create_memory_item(self, role: str, content: str, embedding: np.ndarray,
                            turn: int, conversation_type: str) -> Dict:
        """Create a memory item."""
        return {
            "id": len(self.memory_items),
            "role": role,
            "content": content,
            "embedding": embedding,
            "turn": turn,
            "timestamp": datetime.now().isoformat(),
            "type": conversation_type,
            "importance": self._calculate_importance(content, role, conversation_type),
            "keywords": self._extract_keywords(content),
            "access_count": 0,
            "last_accessed": datetime.now().isoformat(),
        }

    def _calculate_importance(self, content: str, role: str, conv_type: str) -> float:
        """Calculate the importance score of a memory."""
        score = 0.5

        role_weights = {
            "Moderator": 0.8,
            "Decision Maker": 0.9,
            "Expert": 0.7,
            "Analyst": 0.6,
        }
        score += role_weights.get(role, 0.5) * 0.2

        importance_indicators = {
            "suggestion": 0.3, "decision": 0.4, "conclusion": 0.4, "plan": 0.3,
            "agree": 0.2, "disagree": 0.2, "data": 0.2, "analysis": 0.2,
            "action item": 0.5, "timeline": 0.3, "owner": 0.3,
        }
        for indicator, weight in importance_indicators.items():
            if indicator in content:
                score += weight

        type_weights = {
            "decision": 0.8,
            "action_item": 0.9,
            "analysis": 0.7,
            "question": 0.6,
            "general": 0.5,
        }
        score += type_weights.get(conv_type, 0.5) * 0.2

        return min(score, 1.0)

    def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords (simplified version)."""
        words = content.split()
        keywords = []
        important_words = ["optimize", "suggestion", "plan", "data", "analysis", "decision", "implement", "evaluate"]

        for word in words:
            if word in important_words and len(word) > 1:
                keywords.append(word)

        return list(set(keywords))[:5]

    def _is_duplicate(self, new_embedding: np.ndarray, content: str) -> bool:
        """Check whether the new memory duplicates an existing one."""
        if not self.memory_embeddings:
            return False

        similarities = cosine_similarity([new_embedding], self.memory_embeddings)[0]
        max_similarity = np.max(similarities)

        if max_similarity > self.similarity_threshold:
            most_similar_idx = np.argmax(similarities)
            similar_content = self.memory_items[most_similar_idx]["content"]
            length_ratio = len(content) / len(similar_content)
            if 0.7 < length_ratio < 1.3:
                return True

        return False

    def _update_index(self, memory_item: Dict):
        """Update the memory index."""
        if memory_item["role"] not in self.memory_index:
            self.memory_index[memory_item["role"]] = []
        self.memory_index[memory_item["role"]].append(memory_item["id"])

        if memory_item["type"] not in self.memory_index:
            self.memory_index[memory_item["type"]] = []
        self.memory_index[memory_item["type"]].append(memory_item["id"])

        for keyword in memory_item["keywords"]:
            key = f"keyword_{keyword}"
            if key not in self.memory_index:
                self.memory_index[key] = []
            self.memory_index[key].append(memory_item["id"])

    def _prune_memories(self):
        """Prune memories by removing the least important ones."""
        if len(self.memory_items) <= self.max_memories:
            return

        scores = self._calculate_memory_scores()
        remove_count = len(self.memory_items) - self.max_memories
        remove_indices = np.argsort(scores)[:remove_count]

        for idx in sorted(remove_indices, reverse=True):
            self.memory_embeddings.pop(idx)
            self.memory_items.pop(idx)

        self._rebuild_index()

    def _calculate_memory_scores(self) -> List[float]:
        """Calculate memory scores."""
        scores = []
        current_time = datetime.now()

        for item in self.memory_items:
            time_diff = (current_time - datetime.fromisoformat(item["timestamp"])).days
            time_score = max(0, 1 - time_diff * 0.1)
            access_score = min(1.0, item["access_count"] * self.ACCESS_COUNT_FACTOR)
            total_score = (item["importance"] * 0.5 +
                           time_score * 0.3 +
                           access_score * 0.2)
            scores.append(total_score)

        return scores

    def _rebuild_index(self):
        """Rebuild the index from scratch."""
        self.memory_index = {}
        for i, item in enumerate(self.memory_items):
            self._update_index(item)

    def search_memories(self, query: str, role_filter: Optional[str] = None,
                        type_filter: Optional[str] = None,
                        max_results: int = 10) -> List[Dict]:
        """Search for relevant memories."""
        if not self.memory_items:
            return []

        # Degraded mode: return the most recent items in reverse chronological order, no semantic ranking
        if not self.enabled:
            candidates = list(self.memory_items)
            if role_filter:
                candidates = [m for m in candidates if m["role"] == role_filter]
            if type_filter:
                candidates = [m for m in candidates if m["type"] == type_filter]
            return candidates[-max_results:][::-1]

        query_embedding = self.model.encode(query)
        similarities = cosine_similarity([query_embedding], self.memory_embeddings)[0]

        filtered_indices = list(range(len(self.memory_items)))

        if role_filter:
            filtered_indices = [i for i in filtered_indices
                                if self.memory_items[i]["role"] == role_filter]

        if type_filter:
            filtered_indices = [i for i in filtered_indices
                                if self.memory_items[i]["type"] == type_filter]

        scored_memories = self._score_filtered_memories(filtered_indices, similarities)
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        self._update_access_stats(scored_memories[:max_results])

        return [memory for score, memory in scored_memories[:max_results]]

    def _score_filtered_memories(self, filtered_indices: List[int],
                                 similarities: np.ndarray) -> List[tuple]:
        """Score the filtered memories."""
        scored_memories = []
        current_time = datetime.now()

        for idx in filtered_indices:
            memory = self.memory_items[idx]
            time_diff = (current_time - datetime.fromisoformat(memory["timestamp"])).days
            time_factor = max(0.1, 1 - time_diff * self.TIME_DECAY_FACTOR)
            relevance_score = (similarities[idx] * 0.6 +
                               memory["importance"] * 0.3 +
                               time_factor * 0.1)
            scored_memories.append((relevance_score, memory))

        return scored_memories

    def _update_access_stats(self, top_memories: List[tuple]):
        """Update access statistics."""
        for score, memory in top_memories:
            memory["access_count"] += 1
            memory["last_accessed"] = datetime.now().isoformat()

    def get_context_for_agent(self, current_topic: str, agent_role: str,
                              current_turn: int, context_type: str = "discussion") -> str:
        """Generate optimized context for an agent."""
        query = self._build_search_query(current_topic, context_type)
        relevant_memories = self.search_memories(
            query=query,
            role_filter=None,
            max_results=8,
        )
        context_parts = self._format_context_parts(relevant_memories, agent_role, current_turn)
        return "\n".join(context_parts)

    def _build_search_query(self, current_topic: str, context_type: str) -> str:
        """Build a search query."""
        if context_type == "discussion":
            return f"{current_topic} discussion views suggestions"
        if context_type == "decision":
            return f"{current_topic} decision plan action item"
        if context_type == "analysis":
            return f"{current_topic} data analysis evaluation"
        return current_topic

    def _format_context_parts(self, relevant_memories: List[Dict],
                              agent_role: str, current_turn: int) -> List[str]:
        """Format context parts."""
        context_parts = []

        if relevant_memories:
            context_parts.append("## Relevant Discussion History")
            context_parts.extend(self._format_memory_groups(relevant_memories))
        else:
            context_parts.append("## Discussion History")
            context_parts.append("This is the start of the discussion; no related history yet.")

        context_parts.append(f"\n## Current Progress")
        context_parts.append(f"Current round: Round {current_turn}")
        context_parts.append(f"Your role: {agent_role}")

        return context_parts

    def _format_memory_groups(self, relevant_memories: List[Dict]) -> List[str]:
        """Format memory groups."""
        context_parts = []

        decisions = [m for m in relevant_memories if m["type"] == "decision"]
        analyses = [m for m in relevant_memories if m["type"] == "analysis"]
        actions = [m for m in relevant_memories if m["type"] == "action_item"]
        general = [m for m in relevant_memories if m["type"] == "general"]

        if decisions:
            context_parts.append("### Key Decisions")
            for memory in decisions[:2]:
                context_parts.append(f"- {memory['role']}: {memory['content'][:200]}...")

        if analyses:
            context_parts.append("### Analytical Views")
            for memory in analyses[:2]:
                context_parts.append(f"- {memory['role']}: {memory['content'][:200]}...")

        if actions:
            context_parts.append("### Action Items")
            for memory in actions[:2]:
                context_parts.append(f"- {memory['role']}: {memory['content'][:200]}...")

        if general:
            context_parts.append("### Other Discussions")
            for memory in general[:2]:
                context_parts.append(f"- {memory['role']}: {memory['content'][:200]}...")

        return context_parts
