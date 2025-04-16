# tools.py
from urllib.parse import quote
from langchain.tools import Tool
import requests
import json
from dotenv import load_dotenv
import os
from rapidfuzz import fuzz
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model_client = OpenAI(api_key=OPENAI_API_KEY)

# Load valid options from files
def load_options_from_file(filename):
    with open(filename, 'r') as file:
        return [line.strip() for line in file]

# Try to load the valid options
try:
    valid_groups = load_options_from_file('../groups.txt')
    valid_categories = load_options_from_file('../categories.txt')
    valid_subjects = load_options_from_file('../subjects.txt')
except FileNotFoundError as e:
    print(f"Warning: Could not load options file: {e}")
    valid_groups = []
    valid_categories = []
    valid_subjects = []

def filter_candidates(query: str, candidates: list, top_n: int = 10) -> list:
    """
    Use fuzzy string matching to choose the top_n candidate strings from candidates
    that best match the query.
    """
    # Compute a similarity score for each candidate
    scored = [(candidate, fuzz.token_set_ratio(query, candidate)) for candidate in candidates]
    # Sort candidates by score descending
    scored.sort(key=lambda x: x[1], reverse=True)
    # Return the top_n candidates; if no candidates are good matches, return an empty list.
    return [candidate for candidate, score in scored[:top_n]]

def load_valid_values(filename: str) -> list:
    with open(filename, "r", encoding="utf8") as f:
        # Remove empty lines and strip whitespace
        return [line.strip() for line in f if line.strip()]

def load_valid_groups():
    return load_valid_values("../groups.txt")

def load_valid_categories():
    return load_valid_values("../categories.txt")

def llm_map_prompt_to_filters(prompt: str):
    """
    Uses an LLM to map a natural language prompt to valid groups and categories.
    The LLM receives reduced candidate lists (using fuzzy matching) from the full .txt files 
    and is instructed to return a JSON object with the chosen groups and categories.
    
    Expected JSON output format:
       {"groups": ["Group1", "Group2"], "categories": ["Category1", "Category2"]}
    If no match is found for one field, return ["All"] for that field.
    """
    # Load full lists from files
    valid_groups = load_valid_groups()
    valid_categories = load_valid_categories()

    # Pre-filter the lists using fuzzy matching to reduce tokens
    filtered_groups = filter_candidates(prompt, valid_groups, top_n=10)
    filtered_categories = filter_candidates(prompt, valid_categories, top_n=10)
    
    print("Filtered groups:", filtered_groups)
    print("Filtered categories:", filtered_categories)
    # If filtering returns an empty list, default to ["All"]
    if not filtered_groups:
        filtered_groups = ["All"]
    if not filtered_categories:
        filtered_categories = ["All"]

    # Compose the system prompt as before
    system_prompt = (
        "You are an expert at mapping natural language input to valid filter values. "
        "I will provide you a list of valid groups and valid categories along with a user query. "
        "You must choose only values from the provided lists. If none of the items match, "
        "return ['All'] for that field. Return only valid JSON with keys 'groups' and 'categories'."
    )
    
    # Compose the user prompt with only the reduced lists
    user_prompt = (
        f"Valid groups: {json.dumps(filtered_groups)}\n"
        f"Valid categories: {json.dumps(filtered_categories)}\n"
        f"User query: \"{prompt}\"\n\n"
        "Based on the user query, please select the most relevant groups and categories from the lists above. "
        "Return a JSON object with the keys 'groups' and 'categories'."
    )

    try:
        response = model_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0  # Low temperature to keep the output deterministic
        )
        response = response.model_dump()
        answer = response['choices'][0]['message']['content']
        # Parse the response as JSON. If parsing fails, default to ['All'].
        data = json.loads(answer)
        groups = data.get("groups", ["All"])
        categories = data.get("categories", ["All"])
    except Exception as e:
        print(f"LLM mapping failed: {str(e)}")
        groups = ["All"]
        categories = ["All"]

    return groups, categories

def events_from_duke_api(feed_type: str = "json",
                             future_days: int = 45,
                             groups: list = ['All'],
                             categories: list = ['All'],
                             filter_method_group: bool = True,
                             filter_method_category: bool = True) -> str:
    """
    Fetch events from Duke University's public calendar API with optional filters.

    Parameters:
        feed_type (str): Format of the returned data. Acceptable values include:
                         'rss', 'js', 'ics', 'csv', 'json', 'jsonp'. Defaults to 'json'.
        future_days (int): Number of days into the future for which to fetch events.
                           Defaults to 45.
        groups (list):  The organizer or host groups of the events or the related groups in events. For example,
                        '+DataScience (+DS)' refers to events hosted by the DataScience program.
                        Use 'All' to include events from all groups. 
        categories (list): 
                        The thematic or topical category of the events. For example,
                        'Academic Calendar Dates', 'Alumni/Reunion', or 'Artificial Intelligence'.
                         Use 'All' to include events from all categories.
        filter_method_group (bool): 
            - False: Event must match ALL specified groups (AND).
            - True: Event may match ANY of the specified groups (OR).
        filter_method_category (bool): 
            - False: Event must match ALL specified categories (AND).
            - True: Event may match ANY of the specified categories (OR).

    Returns:
        str: Raw calendar data (e.g., in JSON, XML, or ICS format) or an error message.
    """
    
    # When feed_type is not one of these types, add the simple feed_type parameter.
    feed_type_param = ""
    if feed_type not in ['rss', 'js', 'ics', 'csv']:
        feed_type_param = "feed_type=simple"
    
    feed_type_url = feed_type_param if feed_type_param else ""

    if filter_method_group:
        if 'All' in groups:
            group_url = ""
        else:
            group_url = ""
            for group in groups:
                group_url+='&gfu[]='+quote(group, safe="")
    else:
        if 'All' in groups:
            group_url = ""
        else:
            group_url = "&gf[]=" + quote(groups[0], safe="")
            for group in groups[1:]:
                group_url += "&gf[]=" + quote(group, safe="")

    if filter_method_category:
        if 'All' in categories:
            category_url = ""
        else:
            category_url = ""
            for category in categories:
                category_url += '&cfu[]=' + quote(category, safe="")
    else:
        if 'All' in categories:
            category_url = ""
        else:
            category_url = ""
            for category in categories:
                category_url += "&cf[]=" + quote(category, safe="")

    url = f'https://calendar.duke.edu/events/index.{feed_type}?{category_url}{group_url}&future_days={future_days}&{feed_type_url}'

    response = requests.get(url)

    if response.status_code == 200:
        return response.text[:10000]
    else:
        return f"Failed to fetch data: {response.status_code}"
    
def get_events_from_duke_api(prompt: str,
                                   feed_type: str = "json",
                                   future_days: int = 45,
                                   filter_method_group: bool = True,
                                   filter_method_category: bool = True) -> str:
    """
    Retrieve events from Duke University's public calendar API based on a natural language prompt.

    prompt (str): Natural language prompt describing the query for events.

    feed_type (str): Format of the returned data. Acceptable values include:
                         'rss', 'js', 'ics', 'csv', 'json', 'jsonp'. Defaults to 'json'.

    future_days (int): Number of days into the future for which to fetch events.
                           Defaults to 45.

    filter_method_group (bool): 
    - False: Event must match ALL specified groups (AND).
    - True: Event may match ANY of the specified groups (OR).

    filter_method_category (bool): 
    - False: Event must match ALL specified categories (AND).
    - True: Event may match ANY of the specified categories (OR).
    """
    # Use the LLM-based mapping to get groups and categories
    groups, categories = llm_map_prompt_to_filters(prompt)
    
    print(f"LLM mapped prompt '{prompt}' to groups {groups} and categories {categories}")
    
    # Call the original Duke API tool with the determined filters
    return events_from_duke_api(
        feed_type=feed_type,
        future_days=future_days,
        groups=groups,
        categories=categories,
        filter_method_group=filter_method_group,
        filter_method_category=filter_method_category
    )
    

def get_curriculum_with_subject_from_duke_api(subject: str):
    """
    Retrieve curriculum information from Duke University's API by specifying a subject code.
    Returns information about available courses.
    """
    subject_url = quote(subject, safe="")
    url = f'https://streamer.oit.duke.edu/curriculum/courses/subject/{subject_url}?access_token=19d3636f71c152dd13840724a8a48074'
    
    response = requests.get(url)
    
    if response.status_code == 200:
        try:
            # Parse the JSON response
            data = json.loads(response.text)
            
            # Limit the number of courses returned (e.g., first 5)
            if isinstance(data, list) and len(data) > 5:
                limited_data = data[:5]
                # Add a note about limiting the results
                limited_response = {
                    "courses": limited_data,
                    "note": f"Showing 5 out of {len(data)} courses. Use more specific queries to refine results."
                }
                return json.dumps(limited_response)
            else:
                return response.text[:10000]
        except json.JSONDecodeError:
            return "Error: Could not parse API response"
    else:
        return f"Failed to fetch data: {response.status_code}"
    
def get_detailed_course_information_from_duke_api(course_id: str, course_offer_number: str):
    """
    Retrieve curriculum information from Duke University's API by specifying a course ID and course offer number, allowing you to access detailed information about a specific course.

    Parameters:
        course_id (str): The course ID to get curriculum data for. For example, the course ID is 029248' for General African American Studies.
        course_offer_number (str): The course offer number to get curriculum data for. For example, the course offer number is '1' for General African American Studies.

    Returns:
        str: Raw curriculum data in JSON format or an error message.
    """

    url = f'https://streamer.oit.duke.edu/curriculum/courses/crse_id/{course_id}/crse_offer_nbr/{course_offer_number}?access_token=19d3636f71c152dd13840724a8a48074'
    response = requests.get(url)

    if response.status_code == 200:
        return response.text
    else:
        return f"Failed to fetch data: {response.status_code}"
    
def get_people_information_from_duke_api(name: str):
    """
    Retrieve people information from Duke University's API by specifying a name, allowing you to access detailed information about a specific person.

    Parameters:
        name (str): The name to get people data for. For example, the name is 'John Doe'.

    Returns:
        str: Raw people data in JSON format or an error message.
    """

    name_url = quote(name, safe="")

    url = f'https://streamer.oit.duke.edu/ldap/people?q={name_url}&access_token=19d3636f71c152dd13840724a8a48074'

    response = requests.get(url)

    if response.status_code == 200:
        return response.text
    else:
        return f"Failed to fetch data: {response.status_code}"

# New search functions for format compatibility
def search_subject_by_code(query):
    """
    Search for subjects matching a code or description.
    
    Parameters:
        query (str): The search term to look for in subject codes or descriptions.
        
    Returns:
        str: JSON string containing matching subjects.
    """
    # Search by code (like "AIPI" or "CS")
    code_matches = []
    for subject in valid_subjects:
        parts = subject.split(' - ')
        if len(parts) >= 2:
            code = parts[0].strip()
            # Look for the query in the code part
            if query.lower() in code.lower() or query.lower().replace(' ', '') in code.lower().replace('-', '').replace(' ', ''):
                code_matches.append(subject)
    
    # Search by name/description (like "computer science" or "artificial intelligence")
    name_matches = []
    for subject in valid_subjects:
        parts = subject.split(' - ')
        if len(parts) >= 2:
            name = parts[1].strip()
            # Look for the query in the name part
            if query.lower() in name.lower():
                name_matches.append(subject)
    
    # Combine results with code matches first (removing duplicates)
    all_matches = code_matches + [m for m in name_matches if m not in code_matches]
    
    return json.dumps({
        "query": query,
        "matches": all_matches[:5]  # Limit to top 5 matches
    })

def search_group_format(query):
    """
    Search for groups matching a query string.
    
    Parameters:
        query (str): The search term to look for in group names.
        
    Returns:
        str: JSON string containing matching groups.
    """
    matches = [g for g in valid_groups if query.lower() in g.lower()]
    
    return json.dumps({
        "query": query,
        "matches": matches[:5]  # Limit to top 5 matches
    })

def search_category_format(query):
    """
    Search for categories matching a query string.
    
    Parameters:
        query (str): The search term to look for in category names.
        
    Returns:
        str: JSON string containing matching categories.
    """
    matches = [c for c in valid_categories if query.lower() in c.lower()]
    
    return json.dumps({
        "query": query,
        "matches": matches[:5]  # Limit to top 5 matches
    })

def get_pratt_info_from_serpapi(query="Duke Pratt School of Engineering", api_key=None, filter_domain=True):
     """
     Retrieve information about Duke's Pratt School of Engineering using SerpAPI.
     """
     if api_key is None:
         api_key = os.environ.get("SERPAPI_API_KEY")
         if not api_key:
             return json.dumps({"error": "SerpAPI key not found. Please provide an API key or set SERPAPI_API_KEY environment variable."})
     
     # Ensure the query includes Duke Pratt
     if "duke pratt" not in query.lower():
         query = f"Duke Pratt School of Engineering {query}"
     
     # Construct the SerpAPI URL with the query
     encoded_query = quote(query)
     url = f"https://serpapi.com/search.json?q={encoded_query}&engine=google&num=10&api_key={api_key}"
     
     try:
         # Make the request to SerpAPI
         response = requests.get(url, timeout=15)
         response.raise_for_status()
         
         search_results = response.json()
         
         processed_results = process_serpapi_results(search_results, filter_domain)
         
         return json.dumps(processed_results)
         
     except requests.exceptions.RequestException as e:
         return json.dumps({"error": f"Failed to fetch data from SerpAPI: {str(e)}"})
     except json.JSONDecodeError:
         return json.dumps({"error": "Failed to parse SerpAPI response as JSON"})
 
def process_serpapi_results(search_results, filter_domain=True):
     """
     Process and filter SerpAPI results to extract the most relevant information.
     """
     processed_data = {
         "search_metadata": {},
         "organic_results": [],
         "knowledge_graph": {},
         "related_questions": []
     }
     
     # Extract search metadata
     if "search_metadata" in search_results:
         processed_data["search_metadata"] = {
             "query": search_results["search_metadata"].get("query", ""),
             "total_results": search_results.get("search_information", {}).get("total_results", 0)
         }
     
     # Extract organic results
     if "organic_results" in search_results:
         organic_results = search_results["organic_results"]
         
         # Filter for duke.edu domains if requested
         if filter_domain:
             # More aggressive filtering - require "duke" in the link or snippet
             filtered_results = [result for result in organic_results 
                                if "duke" in result.get("link", "").lower() or 
                                   "duke" in result.get("snippet", "").lower()]
             
             # Further prioritize pratt.duke.edu results
             pratt_results = [result for result in filtered_results 
                             if "pratt.duke.edu" in result.get("link", "")]
             
             other_duke_results = [result for result in filtered_results 
                                  if "pratt.duke.edu" not in result.get("link", "")]
             
             # Combine with pratt results first, then other duke results
             processed_results = pratt_results + other_duke_results
             
             # If we have no results after filtering, use the original results
             if not processed_results and organic_results:
                 processed_results = organic_results[:5]  # Just take the top 5
         else:
             processed_results = organic_results
         
         # Extract the most useful information from each result
         for result in processed_results[:8]:  # Limit to top 8 results
             processed_data["organic_results"].append({
                 "title": result.get("title", ""),
                 "link": result.get("link", ""),
                 "snippet": result.get("snippet", ""),
                 "source": result.get("source", "")
             })
     
     # Extract knowledge graph information if available
     if "knowledge_graph" in search_results:
         kg = search_results["knowledge_graph"]
         processed_data["knowledge_graph"] = {
             "title": kg.get("title", ""),
             "type": kg.get("type", ""),
             "description": kg.get("description", ""),
             "website": kg.get("website", ""),
             "address": kg.get("address", "")
         }
     
     # Extract related questions if available
     if "related_questions" in search_results:
         for question in search_results["related_questions"][:4]:  # Limit to top 4 questions
             processed_data["related_questions"].append({
                 "question": question.get("question", ""),
                 "answer": question.get("answer", "")
             })
     
     return processed_data
 
def get_specific_pratt_info(topic="general", subtopic=None, api_key="9339dbe03e129628964af59694c4709f334ee7bf84e7c0c1e335cbc9ea0bbaf6"):
     """
     Retrieve specific information about Duke's Pratt School of Engineering using SerpAPI.
     """
     # Map topics to specific search queries
     topic_queries = {
         "general": "Duke Pratt School of Engineering overview information",
         "academics": "Duke Pratt School of Engineering academic programs degrees majors",
         "admissions": "Duke Pratt School of Engineering admissions requirements application deadlines",
         "ai_meng": "Duke Pratt AI for Product Innovation MEng program curriculum courses",
         "student_life": "Duke Pratt School of Engineering student life experience campus",
         "research": "Duke Pratt School of Engineering research areas labs projects",
         "faculty": "Duke Pratt School of Engineering faculty professors researchers",
         "events": "Duke Pratt School of Engineering events workshops seminars"
     }
     
     # Map subtopics for more specific queries
     subtopic_queries = {
         "academics": {
             "undergraduate": "Duke Pratt School of Engineering undergraduate programs BSE degrees majors",
             "graduate": "Duke Pratt School of Engineering graduate programs masters PhD",
             "courses": "Duke Pratt School of Engineering course offerings classes",
             "requirements": "Duke Pratt School of Engineering degree requirements curriculum"
         },
         "admissions": {
             "undergraduate": "Duke Pratt School of Engineering undergraduate admissions requirements deadlines",
             "graduate": "Duke Pratt School of Engineering graduate admissions requirements deadlines",
             "deadlines": "Duke Pratt School of Engineering application deadlines",
             "requirements": "Duke Pratt School of Engineering application requirements"
         },
         "ai_meng": {
             "curriculum": "Duke Pratt AI for Product Innovation MEng program curriculum courses",
             "admissions": "Duke Pratt AI for Product Innovation MEng program admissions requirements",
             "careers": "Duke Pratt AI for Product Innovation MEng program career outcomes jobs",
             "faculty": "Duke Pratt AI for Product Innovation MEng program faculty instructors"
         }
     }
     
     # Check if the topic is valid
     if topic not in topic_queries:
         return json.dumps({
             "error": f"Topic '{topic}' not found",
             "available_topics": list(topic_queries.keys())
         })
     
     # Construct the query based on topic and subtopic
     if subtopic and topic in subtopic_queries and subtopic in subtopic_queries[topic]:
         query = subtopic_queries[topic][subtopic]
     else:
         query = topic_queries[topic]
     
     # Call the SerpAPI search function
     return get_pratt_info_from_serpapi(query, api_key)