from typing import Dict, Optional
import re
import json
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage

GENERATE_TITLE_PROMPT = \
'''Summarize the content and topics of the following text in one sentence as the title for it.
Do **not** include any additional text or explanations outside of the summary.
'''

EXTRACT_KEYWORDS_PROMPT2 = \
"""
You are Neko. You extracts relevant keywords and their relations to the given message block. Follow these steps:

1. Extract Keywords:
    - Identify keywords that are specific and meaningful, such as:
        - Names (e.g., people, organizations, places)
        - Terms (e.g., technology, medicine, methods)
        - Dates (e.g., specific events, deadlines)
        - Links (e.g., URLs, references)
        - Facts (e.g., concrete statements, claims)
    - Do not include:
        - Overly common words (e.g., "the", "and", "is")
        - Temporary information (e.g., moods, greetings, filler phrases)
        - Neko, which is your name.
        - Zhu Ren, which is how you call the user.

2. Describe Relations:
    - For each keyword, summarize how it is described or mentioned in the message block.
    - Be concise but specific, capturing the essence of how the keyword is used in the context of the message.
    - Avoid using vague expressions e.g. "I", "You", "He", "They".

3. Output Format:
    - Return a JSON object with the following structure:
    {{
        "keyword1": "keyword1: how the keyword1 is described or mentioned in the message block",
        "keyword2": "keyword2: how the keyword2 is described or mentioned in the message block"
    }}

4. Example:
    - Input: "John Doe will present the Q2 financial report on July 15th, 2023. The report shows a revenue increase of 15 percent compared to last year."
    - Output:
    {{
        "John Doe": "John Doe: will present the Q2 financial report"
        "Q2 financial report": "The Q2 financial report: shows a revenue increase of 15 percent compared to last year"
        "July 15th, 2023": "July 15th, 2023: date when John Doe will present the Q2 financial report"
        "15 percent revenue increase": "15 percent revenue increase: the result that the Q2 financial report shows"
    }}

Now, process the following message block. Do **not** include any additional text or explanations outside of the dictionary.
"""

EXTRACT_USER_QUERY_KEYWORDS_PROMPT = \
'''You are Neko. You need to parse user queries and generates subqueries ot get relevant knowledge required to respond to the user. Follow these steps:

2. **Generate Subqueries**:
   - For each aspect, generate a subquery that describes what knowledge you want to learn.
   - Ensure that these subqueries cover all knowledge you need to learn to respond to the user queries.

1. **Identify Key Components**:
   - For each subquery, identify one keyword (e.g., the entity, subject, time, location, name that appears).

3. **Output Format**:
   - Return a JSON object with the following structure:
    {{
        "key component 1": "subquery 1",
        "key component 2": "subquery 2"
    }}

Do **not** include any additional text or explanations outside of the dictionary.
'''


def generate_title(llm, text:str, system_prompt=GENERATE_TITLE_PROMPT) -> [AIMessage, str]:
    '''
    Generate one title for the text, and return [AIMessage, the title].
    '''
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", "{text}"),
        ]
    )

    prompt = prompt_template.invoke({"text": text})
    output = llm.invoke(prompt)
    return output, output.content

def _generate_summary_dict(llm, text:str, system_prompt) -> [AIMessage, Optional[Dict[str, str]]]:
    """
    Generate the summary of the text in the format of Dict[keyword, desc].
    
    First return: 
        raw output of LLM
    Second return:
        keyword: the keyword of the text extracted by the LLM
        desc: the description of the keyword and its context in the text
    """

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", "{text}"),
        ]
    )

    prompt = prompt_template.invoke({"text": text})
    output = llm.invoke(prompt)

    def parse_to_dict(output:str)->Optional[Dict[str, str]]:
        # Check if the string ends with a closing brace, if not, append it
        # if not output.strip().endswith('}'):
        #     output = output.strip() + '}'

        # Regular expression to match the JSON-like dictionary
        json_match = re.search(r'(\{.*\})', output, re.DOTALL)

        if json_match:
            json_str = json_match.group(1)
            try:
                # Parse the extracted JSON string into a dictionary
                parsed_dict = json.loads(json_str)
                return parsed_dict
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
        else:
            print("No JSON-like dictionary found in the output.")
        return None
    
    rst = parse_to_dict(output.content)
    return output, rst

def generate_summary_dict(llm, text:str, system_prompt=EXTRACT_KEYWORDS_PROMPT2, max_retry=3) -> [Optional[AIMessage], Optional[Dict[str, str]]]:
    """
    Generate the summary of the text in the format of Dict[keyword, desc].
    
    First return: 
        raw output of LLM (AI Message)
    Second return:
        keyword: the keyword of the text extracted by the LLM
        desc: the description of the keyword and its context in the text
    """
    cnt = 0
    while(cnt<max_retry):
        output, rst = _generate_summary_dict(llm, text, system_prompt)
        if rst:  # Successfully parsed and return.
            return output, rst
        else:
            print("Parsing the LLM output to Dict failed:")
            print(output.content)
            print(f"Retrying {cnt}/{max_retry}...")
            cnt+=1
    return None, None 


