# tools.py
from urllib.parse import quote
from langchain.tools import Tool
import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()

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

def get_events_from_duke_api(feed_type: str = "json",
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
        return response.text
    else:
        return f"Failed to fetch data: {response.status_code}"
    

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
                return response.text
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

# Create tools for LangChain
tools = [
    Tool(
        name="get_duke_events",
        func=get_events_from_duke_api,
        description=(
            "Use this tool to retrieve upcoming events from Duke University's calendar via Duke's public API. "
            "IMPORTANT: 'groups' parameter values must be from groups.txt list. "
            "IMPORTANT: 'categories' parameter values must be from categories.txt list. "
            "Parameters:"
            "   feed_type (str): Format of the returned data. Acceptable values include 'rss', 'js', 'ics', 'csv', 'json', 'jsonp'."
            "   future_days (int): Number of days into the future for which to fetch events. Defaults to 45."
            "   groups (list): List of groups to filter events by. Use ['All'] to include events from all groups."
            "   categories (list): List of categories to filter events by. Use ['All'] to include events from all categories."
            "   filter_method_group (bool): True: Event may match ANY of the specified groups (OR). False: Event must match ALL specified groups (AND)."
            "   filter_method_category (bool): True: Event may match ANY of the specified categories (OR). False: Event must match ALL specified categories (AND)."
        )
    ),
    Tool(
        name="get_curriculum_with_subject_from_duke_api",
        func=get_curriculum_with_subject_from_duke_api,
        description=(
            "Use this tool to retrieve curriculum information from Duke University's API."
            "IMPORTANT: The 'subject' parameter must be from subjects.txt list. "
            "Parameters:"
            "   subject (str): The subject to get curriculum data for. For example, the subject is 'ARABIC-Arabic'."
            "Return:"
            "   str: Raw curriculum data in JSON format or an error message. If valid result, the response will contain each course's course id and course offer number for further queries."
        )
    ),
    Tool(
        name="get_detailed_course_information_from_duke_api",
        func=get_detailed_course_information_from_duke_api,
        description=(
            "Use this tool to retrieve detailed curriculum information from Duke University's API."
            "The course ID and course offer number can be obtained from get_curriculum_with_subject_from_duke_api."
            "Parameters:"
            "   course_id (str): The course ID to get curriculum data for. For example, the course ID is 029248' for General African American Studies."
            "   course_offer_number (str): The course offer number to get curriculum data for. For example, the course offer number is '1' for General African American Studies."
            "Return:"
            "   str: Raw curriculum data in JSON format or an error message."
        )
    ),
    Tool(
        name="get_people_information_from_duke_api",
        func=get_people_information_from_duke_api,
        description=(
            "Use this tool to retrieve people information from Duke University's API."
            "Parameters:"
            "   name (str): The name to get people data for. For example, the name is 'Brinnae Bent'."
            "Return:"
            "   str: Raw people data in JSON format or an error message."
        )
    ),
    Tool(
        name="search_subject_by_code",
        func=search_subject_by_code,
        description=(
            "Use this tool to find the correct format of a subject before using get_curriculum_with_subject_from_duke_api. "
            "This tool handles case-insensitive matching and partial matches. "
            "Example: 'cs' might return 'COMPSCI - Computer Science'. "
            "Always use this tool first if you're uncertain about the exact subject format."
        )
    ),
    Tool(
        name="search_group_format",
        func=search_group_format,
        description=(
            "Use this tool to find the correct format of a group before using get_events_from_duke_api. "
            "This tool handles case-insensitive matching and partial matches. "
            "Example: 'data science' might return '+DataScience (+DS)'. "
            "Always use this tool first if you're uncertain about the exact group format."
        )
    ),
    Tool(
        name="search_category_format",
        func=search_category_format,
        description=(
            "Use this tool to find the correct format of a category before using get_events_from_duke_api. "
            "This tool handles case-insensitive matching and partial matches. "
            "Example: 'ai' might return 'Artificial Intelligence'. "
            "Always use this tool first if you're uncertain about the exact category format."
        )
    ),
]