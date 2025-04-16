# agent.py
from langchain.agents import initialize_agent, AgentType
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.tools import Tool
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
import os
import json
from dotenv import load_dotenv

# Import your custom tools from tools.py
from tools import (
    get_events_from_duke_api,
    get_curriculum_with_subject_from_duke_api,
    get_detailed_course_information_from_duke_api,
    get_people_information_from_duke_api,
    search_subject_by_code,
    search_group_format,
    search_category_format
)

# Load environment variables from .env file
load_dotenv()

def create_duke_agent():
    """
    Create a LangChain agent with the Duke tools.
    API keys are loaded from .env file.
    
    Returns:
        An initialized LangChain agent
    """
    # Get API keys from environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    # Check if API keys are available
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    # Define the tools
    tools = [
        Tool(
            name="get_duke_events",
            func=get_events_from_duke_api,
            description=(
                "Use this tool to retrieve upcoming events from Duke University's calendar. "
                "IMPORTANT: This tool requires exact format for groups and categories parameters. "
                "You should first use search_group_format and search_category_format to find correct formats."
                "Parameters: feed_type (str), future_days (int), groups (list), categories (list), "
                "filter_method_group (bool), filter_method_category (bool)"
            )
        ),
        Tool(
            name="get_curriculum_with_subject_from_duke_api",
            func=get_curriculum_with_subject_from_duke_api,
            description=(
                "Use this tool to retrieve curriculum information for a specific subject. "
                "IMPORTANT: This tool requires the exact format for the subject parameter. "
                "You should first use search_subject_by_code to find the correct format."
                "Parameters: subject (str)"
            )
        ),
        Tool(
            name="get_detailed_course_information_from_duke_api",
            func=get_detailed_course_information_from_duke_api,
            description=(
                "Use this tool to retrieve detailed information about a specific course. "
                "Parameters: course_id (str), course_offer_number (str) - both obtained from get_curriculum_with_subject_from_duke_api."
            )
        ),
        Tool(
            name="get_people_information_from_duke_api",
            func=get_people_information_from_duke_api,
            description=(
                "Use this tool to retrieve information about Duke people by specifying a name."
            )
        ),
        Tool(
            name="search_subject_by_code",
            func=search_subject_by_code,
            description=(
                "Use this tool to find the correct format of a subject before using get_curriculum_with_subject_from_duke_api. "
                "Example: 'cs' might return 'COMPSCI - Computer Science'. "
                "Always use this tool first if you're uncertain about the exact subject format."
            )
        ),
        Tool(
            name="search_group_format",
            func=search_group_format,
            description=(
                "Use this tool to find the correct format of a group before using get_duke_events. "
                "Example: 'data science' might return '+DataScience (+DS)'. "
                "Always use this tool first if you're uncertain about the exact group format."
            )
        ),
        Tool(
            name="search_category_format",
            func=search_category_format,
            description=(
                "Use this tool to find the correct format of a category before using get_duke_events. "
                "Example: 'ai' might return 'Artificial Intelligence'. "
                "Always use this tool first if you're uncertain about the exact category format."
            )
        )
    ]
    
    # Create a memory instance
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Initialize the LLM with the OpenAI API key
    llm = ChatOpenAI(
        api_key=openai_api_key,
        model_name="gpt-4",
        temperature=0
    )
    
    # System prompt for agentic search approach
    system_prompt = """
    You are a Duke University assistant with access to specialized Duke API tools. Follow these steps for each query:

    1. THINK: Analyze what information the user is seeking and which tool is appropriate.

    2. FORMAT SEARCH: If the user's query contains subject, group, or category names that may not be in the exact required format:
       - Use search_subject_by_code to find the correct subject format
       - Use search_group_format to find the correct group format
       - Use search_category_format to find the correct category format

    3. ACT: Once you have the correct format, execute the appropriate API call with the correctly formatted parameters.

    4. OBSERVE: Analyze the results returned by the tool.

    5. RESPOND: Provide a clear, helpful response based on the tool's output.

    IMPORTANT:
    - Never call API tools directly with user-provided formats for subjects, groups, or categories
    - Always use the search tools first to find the correct format
    - If multiple possible matches are found, ask the user to clarify which one they want or choose the most likely match
    - When showing results, don't mention format correction unless it's relevant to explain an error

    This agentic approach ensures you'll provide accurate information while handling format variations.
    """
    
    # Create a proper chat prompt template
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    
    # Initialize the agent with the correct prompt
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        max_iterations=5,
        early_stopping_method="generate",
        handle_parsing_errors=True,
        prompt=prompt  # Use the properly formatted prompt
    )
    
    return agent

# Example usage
def process_user_query(query):
    try:
        # Create the agent
        duke_agent = create_duke_agent()
        
        # Process the query using invoke
        response = duke_agent.invoke({"input": query})
        
        # Extract the agent's response
        return response.get("output", "I couldn't process your request at this time.")
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return f"An error occurred: {str(e)}"

# Example usage
def main():
    # Test queries that demonstrate format compatibility
    test_queries = [
        "What events are happening at Duke this week?",
        "Get me detailed information about the AIPI courses",
        "Tell me about Computer Science classes",
        "Are there any AI events at Duke?",
        "What cs courses are available?",
        "Tell me about aipi program",
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        response = process_user_query(query)
        print(f"Response: {response}")
        print("-" * 80)

if __name__ == "__main__":
    main()