from abc import ABC, abstractmethod
from typing import List
from datetime import datetime 

MESSAGE_SEPERATOR="\n<message seperator>\n"

class MemoryNode:
    """
    Base class for all memory-related nodes in the graph.
    """
    def __init__(self, node_id: str, embeddings: List[float] = None, content: str = ""):
        self.node_id = node_id
        self.embeddings = embeddings if embeddings else []
        self.content = content

    def __repr__(self):
        return f"MemoryNode(id={self.node_id}, content={self.content[:30]})"

class KeywordNode(MemoryNode):
    """
    Represents a keyword node in the memory graph.
    Inherits from MemoryNode.
    """
    def __init__(self, keyword_id: str, keyword: str, content:str = "", embeddings: List[float] = None):
        super().__init__(node_id=keyword_id, embeddings=embeddings, content=content)
        self.keyword = keyword

    def __repr__(self):
        return f"KeywordNode(id={self.node_id}, keyword={self.keyword})"


class MessageBlock(MemoryNode):
    """
    Represents a message block in the memory graph.
    Inherits from MemoryNode.
    """
    def __init__(self, block_id: str, messages: List[str], timestamp: str = "", title:str = "", embeddings: List[float] = None):
        content = MESSAGE_SEPERATOR.join(messages)  # Combine messages into a single content string
        super().__init__(node_id=block_id, embeddings=embeddings, content=content)
        self.messages = messages
        if timestamp:
            self.timestamp = timestamp
        else:
            self.timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        self.title = title

    def __repr__(self):
        return f"MessageBlock(id={self.node_id}, messages={self.content}, timestamp={self.timestamp})"



class RelationEdge:
    """
    Represents an edge connecting two nodes in the memory graph.
    type: "describes", "follows", "refers"
    """
    def __init__(self, 
                relation_id: str, 
                source: str, target: str, 
                relation_type: str, relation_desc: str, 
                embeddings: List[float] = None):
        self.relation_id = relation_id
        self.source = source  # ID of the source node
        self.target = target  # ID of the target node
        self.relation_type = relation_type  # E.g., "describes", "follows", "refers"
        self.relation_desc = relation_desc
        self.embeddings = embeddings if embeddings else []

    def __repr__(self):
        return f"RelationEdge(id={self.relation_id}, source={self.source}, target={self.target}, type={self.relation_type}, desc={self.relation_desc})"

    def __eq__(self, other):
        if not isinstance(other, RelationEdge):
            return False
        return self.relation_id == other.relation_id

    def __hash__(self):
        return hash(self.relation_id)


class MemoryManager(ABC):
    @abstractmethod
    def add_message(self, message_block: MessageBlock, keywords: List[KeywordNode], relations: List[RelationEdge]):
        """
        Add a new message block to the memory graph, along with keywords and relations.
        :param message_block: A MessageBlock instance containing content, keywords, and metadata.
        :param keywords: A list of KeywordNode instances related to this message.
        :param relations: A list of RelationEdge instances representing the relationships between the keyword and the message block.
        """
        pass

    @abstractmethod
    def link_messages(self, relation: RelationEdge):
        """
        Link two message blocks or a keyword and a message block with a relation.
        :param relation: A RelationEdge instance representing the relationship.
        """
        pass

    @abstractmethod
    def visualize_graph(self, output_path: str):
        """
        Visualize the graph for debugging or demonstration.
        :param output_path: Path to save the visualization.
        """
        pass

    @abstractmethod
    def get_keyword(self, id: str) -> KeywordNode:
        """
        Retrieve a KeywordNode by its ID.
        :param id: The ID of the keyword to retrieve.
        :return: The corresponding KeywordNode object.
        """
        pass

    @abstractmethod
    def get_message_block(self, id: str) -> MessageBlock:
        """
        Retrieve a MessageBlock by its ID.
        :param id: The ID of the message block to retrieve.
        :return: The corresponding MessageBlock object.
        """
        pass

    @abstractmethod
    def get_relation_by_node(self, node_id: str) -> List[RelationEdge]:
        """
        Retrieve the list of RelationEdge connected to the node.
        :param node_id: The ID of the node to retrieve relations for.
        :return: A list of RelationEdge instances connected to the node.
        """
        pass
