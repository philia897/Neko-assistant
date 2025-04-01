import networkx as nx
from memory_manager import *
from typing import List
from datetime import datetime
import sqlite3

class NetworkXMemoryManager(MemoryManager):
    def __init__(self, db_path: str = "memory.db"):
        self.graph = nx.DiGraph()
        self.db_path = db_path
        self._initialize_database()

    def _initialize_database(self):
        """
        Initializes the SQLite database and creates necessary tables for keywords, message blocks, and relations.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create table for keywords
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS keywords (
                    keyword_id TEXT PRIMARY KEY,
                    content TEXT
                    keyword TEXT NOT NULL,
                    embeddings TEXT
                )
            ''')

            # Create table for message blocks
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS message_blocks (
                    block_id TEXT PRIMARY KEY,
                    messages TEXT NOT NULL,
                    title TEXT,
                    timestamp TEXT NOT NULL,
                    embeddings TEXT
                )
            ''')

            # Create table for relations
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS relations (
                    relation_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    target TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    relation_desc TEXT,
                    embeddings TEXT,
                )
            ''')

    def _serialize_embeddings(self, embeddings: Optional[List[float]]) -> str:
        """
        Serializes a list of embeddings to a string for storage in SQLite.
        """
        return ",".join(map(str, embeddings)) if embeddings else ""

    def _deserialize_embeddings(self, embeddings_str: str) -> List[float]:
        """
        Deserializes a string of embeddings from SQLite into a list of floats.
        """
        return list(map(float, embeddings_str.split(","))) if embeddings_str else []

    def add_keyword(self, keyword: KeywordNode):
        """
        Add a keyword to the database and graph.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR IGNORE INTO keywords (keyword_id, content, keyword, embeddings)
                VALUES (?, ?, ?, ?)
            ''', (
                keyword.node_id,
                keyword.keyword,
                keyword,keyword,
                self._serialize_embeddings(keyword.embeddings)
            ))
            self.graph.add_node(keyword.node_id, node_type="KeywordNode")

    def add_message_block(self, message_block: MessageBlock):
        """
        Add a message block to the database and graph.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR IGNORE INTO message_blocks (block_id, title, messages, timestamp, embeddings)
                VALUES (?, ?, ?, ?)
            ''', (
                message_block.node_id,
                message_block.title,
                message_block.content,
                message_block.timestamp,
                self._serialize_embeddings(message_block.embeddings)
            ))
            self.graph.add_node(message_block.node_id, node_type="MessageBlock")

    def add_relation(self, relation: RelationEdge):
        """
        Add a relation edge to the database and graph.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR IGNORE INTO relations (relation_id, source, target, relation_type, relation_desc, embeddings)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                relation.relation_id,
                relation.source,
                relation.target,
                relation.relation_type,
                relation.relation_desc,
                self._serialize_embeddings(relation.embeddings)
            ))
            self.graph.add_edge(relation.source, relation.target, relation_type=relation.relation_type)

    def link_messages(self, relation: RelationEdge):
        """ Link two message blocks (or a keyword and a message block) with a relation. """
        self.add_relation(relation)

    def add_message(self, message_block: MessageBlock, keywords: List[KeywordNode], relations: List[RelationEdge]):
        """ Add a message block to the graph, along with keywords and relations. """
        # Add the message block node with its metadata
        self.add_message_block(message_block)

        # Add keyword nodes and link them to the message block
        for keyword in keywords:
            if keyword.keyword_id not in self.graph:
                self.add_keyword(keyword)

        # Add the relations between message blocks (or between keyword and message)
        for relation in relations:
            link_messages(relation)

    def visualize_graph(self, output_path: str):
        """ Visualize the graph. """
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_color='lightblue', edge_color='gray')
        labels = nx.get_edge_attributes(self.graph, 'relation_type')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=labels)
        plt.savefig(output_path)
        plt.close()

    def get_keyword(self, keyword_id: str) -> KeywordNode:
        """
        Retrieve a keyword by its ID.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM keywords WHERE keyword_id = ?', (keyword_id,))
            row = cursor.fetchone()
            if row:
                return KeywordNode(
                    keyword_id=row[0],
                    content=row[1],
                    keyword=row[2],
                    embeddings=self._deserialize_embeddings(row[3])
                )
            raise ValueError(f"Keyword with ID {keyword_id} not found.")

    def get_message_block(self, block_id: str) -> MessageBlock:
        """
        Retrieve a message block by its ID.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM message_blocks WHERE block_id = ?', (block_id,))
            row = cursor.fetchone()
            if row:
                return MessageBlock(
                    block_id=row[0],
                    messages=row[1].split(MESSAGE_SEPERATOR),
                    title=row[2],
                    timestamp=row[3],
                    embeddings=self._deserialize_embeddings(row[4])
                )
            raise ValueError(f"Message block with ID {block_id} not found.")

    def get_relation_by_node(self, node_id: str) -> List[RelationEdge]:
        """
        Retrieve all relations connected to a specific node.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM relations WHERE source = ? OR target = ?
            ''', (node_id, node_id))
            rows = cursor.fetchall()
            relations = [
                RelationEdge(
                    relation_id=row[0],
                    source=row[1],
                    target=row[2],
                    relation_type=row[3],
                    relation_desc=row[4],
                    embeddings=self._deserialize_embeddings(row[5])
                )
                for row in rows
            ]
            return relations

if __name__ == "__main__":
    # Create some keywords with metadata (e.g., embeddings)
    kw1 = KeywordNode(keyword_id="kw1", keyword="help", embeddings=[0.1, 0.2, 0.3])
    kw2 = KeywordNode(keyword_id="kw2", keyword="account", embeddings=[0.4, 0.5, 0.6])
    kw3 = KeywordNode(keyword_id="kw3", keyword="assist", embeddings=[0.7, 0.8, 0.9])

    # Create some message blocks
    msg1 = MessageBlock(
        block_id="msg1",
        messages=["Hi, I need your help with my account", "Sure, let me help you with your account"],
        timestamp=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        embeddings=[0.01, 0.02, 0.03]
    )

    msg2 = MessageBlock(
        block_id="msg2",
        messages=["How you will assist me?", "Please give me more details about what is wrong with your account"],
        timestamp=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        embeddings=[0.07, 0.08, 0.09]
    )

    # Create relations (MessageBlock -> MessageBlock or Keyword -> MessageBlock)
    relation1 = RelationEdge(
        source="kw1",
        target="msg1",
        relation_type="describes",
        relation_desc="keyword 'help' is in msg1",
        embeddings=[0.1, 0.2]
    )
    relation2 = RelationEdge(
        source="kw2",
        target="msg1",
        relation_type="describes",
        relation_desc="keyword 'account' is in msg1",
        embeddings=[0.3, 0.4]
    )
    relation3 = RelationEdge(
        source="kw2",
        target="msg2",
        relation_type="describes",
        relation_desc="keyword 'account' is in msg2",
        embeddings=[0.3, 0.4]
    )
    relation4 = RelationEdge(
        source="kw3",
        target="msg2",
        relation_type="describes",
        relation_desc="keyword 'assist' is in msg2",
        embeddings=[0.3, 0.4]
    )
    relation5 = RelationEdge(
        source="msg1",
        target="msg2",
        relation_type="follows",
        relation_desc="",
        embeddings=[0.5, 0.6]
    )

    # Initialize memory manager
    manager = NetworkXMemoryManager()

    # Add messages to memory with associated keywords and relations
    manager.add_message(msg1, keywords=[kw1, kw2], relations=[relation1, relation2])
    manager.add_message(msg2, keywords=[kw2, kw3], relations=[relation3, relation4, relation5])

    # Visualize the graph
    # manager.visualize_graph("graph_output.png")
    print("Graph visualized and saved as graph_output.png")
