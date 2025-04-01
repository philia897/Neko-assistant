from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from memory.sqlite_memory_manager import SQLiteMemoryManager
from memory import MessageBlock, KeywordNode, RelationEdge
from memory.utils import sort_items_by_relevance
from typing import List
import logging
from datetime import datetime 

from llm.utils import generate_summary_dict, EXTRACT_USER_QUERY_KEYWORDS_PROMPT

USER_NAME = "Felix"


PERSONA = f"\
Your name is Neko, a cat girl working as the maid to assist {USER_NAME}. You should always call him Zhu Ren, to fit your maid character. \
For private questions (About memory, chat history, etc), strictly uses only the information provided in the memory. Do not infer, assume, or add any details that are not explicitly stated. If the memory does not contain specific information, respond with \"I don't have enough information to answer that.\"\
"

SYSTEM_PROMPT = \
"""
{persona}

Relevant memory:
{memory}

The chat history:
{chat_history}
"""



def get_logger(log_level=logging.INFO):
    # Create a named logger for your application
    logger = logging.getLogger("MemoryChatbot")
    logger.setLevel(log_level)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    # Set a formatter for the handler
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(console_handler)

    # Prevent other loggers from propagating to the root logger
    logger.propagate = False

    return logger

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

def _choose_ranks(similarities: List[float], ranks: List[int]) -> List[int]:

    # Ensure both lists have the same length
    if len(similarities) != len(ranks):
        raise ValueError("The lengths of similarities and ranks must match.")
    
    # Filter and return ranks based on the similarity threshold
    return [rank for sim, rank in zip(similarities, ranks) if sim > 0.8]

def select_elements(lst: list, indices: list) -> list:
    return [lst[i] for i in indices if 0 <= i < len(lst)]

def retrieve_relevant_memory(query:str, memory_manager, embedding_model, llm):
    _, query_dict = generate_summary_dict(llm, query, EXTRACT_USER_QUERY_KEYWORDS_PROMPT)

    memory = []
    for k,v in query_dict.items():
        logger.debug(f"{k}: {v}")
        query_embedding = embedding_model.embed_query(v)

        keywords = memory_manager.search_keywords(k)
        if len(keywords)==0:
            logger.debug(f"No search result for {k}, semantic searching instead...")
            keywords = memory_manager.search_keywords_by_embedding(query_embedding)
        logger.debug(f"matched keywords: {keywords}")
        relations = []
        for keyword in keywords:
            relations = relations + memory_manager.get_relation_by_node(keyword.node_id)
        if len(relations)==0:
            logger.debug(f"No relations for keyword {k}")
            continue
        relations = list(set(relations)) # remove duplicated relations
        possible_embeddings = [relation.embeddings for relation in relations]
        similarities, ranks = sort_items_by_relevance(query_embedding, possible_embeddings)
        selected_relations = select_elements(relations, ranks[:1])
        logger.debug(f"Sim: {similarities[0]}: {selected_relations[0].relation_desc}")
        for r in selected_relations:
            memory.append(r.relation_desc)

    return "\n".join(memory)

def process_messages_and_save_to_memory(messages:List[str], memory_manager, embedding_model, llm):
    logger.info("")
    block_id = "msg-" + datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    message_block = MessageBlock(block_id=block_id, messages=messages)
    message_block.embeddings = embedding_model.embed_query(message_block.content)

    _, summary_dict = generate_summary_dict(llm, message_block.content)
    keywords = []
    relations = []
    cnt = 0
    for k,v in summary_dict.items():
        matched_keywords = memory_manager.search_keywords(k)
        key = k if len(matched_keywords) == 0 else matched_keywords[0].node_id

        if len(matched_keywords) == 0:
            key_embd = embedding_model.embed_query(v)
            keywords.append(KeywordNode(keyword_id=key, keyword=key, embeddings=key_embd))
        else:
            relations = memory_manager.get_relation_by_node(matched_keywords[0].node_id)
            desc = "\n".join([relation.relation_desc for relation in relations])
            key_embd = embedding_model.embed_query(desc + "\n" + v)
            keywords.append(KeywordNode(keyword_id=key, keyword=key, embeddings=key_embd))

        relation_id = str(cnt) + "-" + block_id
        cnt += 1
        relations.append(RelationEdge(relation_id=relation_id, source=k, target=block_id, 
                            relation_type="describes", relation_desc=v, embeddings=embedding_model.embed_query(v)))
        logger.debug(f"{cnt}: Key: {k} : {v}")
    memory_manager.add_message(message_block=message_block, keywords=keywords, relations=relations)
    return 

class MemoryChatbot:
    def __init__(self, llm, memory_manager, embedding_model, **args):
        """
        Initialize the chatbot with an LLM, memory manager, and embedding model.

        Args:
            llm: The LLM model used for generating responses.
            memory_manager: The memory manager for storing and retrieving chat history.
            embedding_model: The embedding model used for memory-related tasks.
        """
        self.llm = llm
        self.memory_manager = memory_manager
        self.embedding_model = embedding_model
        self.chat_history = []  # Store chat history for the current session.
        self.args = args

    def retrieve_memory(self, query):
        """
        Retrieves relevant memory for a given query.

        Args:
            query (str): The user's query.

        Returns:
            str: Retrieved memory as a text string.
        """
        logger.info("\n\n*********** Start retrieving relevant memory *****************\n\n")
        history = "Chat History:\n" + "\n".join(self.chat_history)
        prompt = history + f"\n{USER_NAME}: " + query
        logger.debug("\n\nNow process the following user query:\n %s", prompt)
        memory = retrieve_relevant_memory(prompt, self.memory_manager, self.embedding_model, self.llm)
        logger.debug("Retrieved memory: %s", memory)
        logger.info("\n\n*********** End retrieving relevant memory *****************\n\n")
        return memory

    def process_and_save_memory(self, messages):
        """
        Processes and saves chat messages to memory.

        Args:
            messages (list of str): The messages to be saved to memory.
        """
        logger.info("\n\n*********** Start processing and saving memory *****************\n\n")
        logger.debug("Start saving the following messages:")
        for msg in messages:
            logger.debug(msg)
        process_messages_and_save_to_memory(messages, self.memory_manager, self.embedding_model, self.llm)
        logger.info("\n\n*********** End processing and saving memory *****************\n\n")

    def stream_response(self, prompt):
        """
        Streams the response generated by the LLM.

        Args:
            prompt (str): The input prompt for the LLM.

        Yields:
            str: Generated tokens one by one.
        """
        for token in self.llm.stream(prompt):  # Ensure the LLM supports streaming
            yield token

    def run_chat_round(self, user_input):
        """
        Executes one round of chat: retrieves memory, generates response, and updates memory.

        Args:
            user_input (str): The user's input.

        Returns:
            str: The chatbot's response.
        """
        # Retrieve relevant memory
        memory = self.retrieve_memory(user_input)

        # Build the prompt with memory and chat history
        history_prompt = ""
        if self.chat_history:
            max_l = self.args.get("history_length", 2)
            if len(self.chat_history) >= max_l:
                self.chat_history = self.chat_history[-max_l:]
            history_prompt += "\n".join(self.chat_history) + "\n\n"

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("user", "{user_input}"),
            ]
        )

        prompt = prompt_template.invoke({"persona": PERSONA, "memory": memory, "chat_history": history_prompt, "user_input": user_input})

        # Stream response
        logger.debug("Generating response for prompt: %s", prompt)
        print()
        response = ""
        print("*********** Neko Reply ****************\n")
        for token in self.stream_response(prompt):
            print(token.content, end="", flush=True)  # Stream the token to the console
            response += token.content
        print()  # Add a new line after streaming is complete.
        print("\n*************** Neko End Reply **************\n\n")
        # Update chat history and memory
        self.chat_history.append(f"{USER_NAME}: {user_input}")
        self.chat_history.append(f"Neko: {response}")
        self.process_and_save_memory([f"{USER_NAME}: {user_input}", f"Neko: {response}"])

        return response

# Example usage
if __name__ == "__main__":
    logger = get_logger(logging.DEBUG)

    # Initialize models and memory manager (replace with actual instances)
    llm = ChatOllama(model="qwen2.5")  # Example LLM
    memory_manager = SQLiteMemoryManager("memory.db")  # Replace with actual memory manager instance
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    chatbot = MemoryChatbot(llm, memory_manager, embedding_model)

    print("Chatbot is ready! Type your message below.")
    while True:
        print("*********** User Input ****************\n")
        user_input = input("You: ")
        print("\n*************** User End Input **************\n")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        timestamp = get_timestamp()
        user_input = f"({timestamp}) {user_input}"
        chatbot.run_chat_round(user_input)
