# AI-Based Movie Recommendation System
# Enhanced version with improved Ollama integration

import requests
import json
import os
import dotenv
import time
from typing import Dict, List, Any, Optional, Union
import sys
import re # Added missing import
import logging # Added logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ANSI color codes for terminal output (kept for potential direct script use, but won't show colors via Flask logs)
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Check if Ollama is running
def check_ollama_status() -> bool:
    """Check if Ollama is running and available"""
    try:
        # First check if Ollama is installed and running
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            logging.info("Ollama is running and available.")
            return True
        else:
            logging.warning(f"Ollama is responding but returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        logging.warning("Ollama connection failed. Make sure Ollama is installed and running.")
        logging.warning("Installation instructions: https://ollama.ai/download")
        logging.warning("After installing, start Ollama and try again.")
        return False
    except requests.exceptions.Timeout:
        logging.warning("Ollama connection timed out. The service might be starting up or overloaded.")
        return False
    except requests.exceptions.RequestException as e:
        logging.error(f"Ollama error: {str(e)}")
        return False

# Function to get movie details from user input (kept for potential direct script use)
def get_movie_input() -> tuple:
    """Get movie title and year from user input"""
    print(f"{Colors.HEADER}{Colors.BOLD}🎬 AI Movie Recommendation System 🎬{Colors.ENDC}")
    print(f"{Colors.CYAN}Let's find movies you'll love based on your taste!{Colors.ENDC}\n")
    
    movie = input(f"{Colors.YELLOW}What's a movie you've watched recently that you liked? {Colors.ENDC}")
    year = input(f"{Colors.YELLOW}What year did '{movie}' come out in? (Press Enter to skip) {Colors.ENDC}")
    
    return movie, year

# Function to get API key from environment variables
def get_api_key() -> Optional[str]:
    """Load and return the TMDB API key from environment variables"""
    dotenv.load_dotenv()
    api_key = os.getenv("TMDB_API_KEY")
    
    if not api_key:
        logging.error("Error: TMDB_API_KEY not found in environment variables.")
        logging.error("Please create a .env file with your TMDB API key in the format: TMDB_API_KEY=your_key_here")
        # Removed print statements
    
    return api_key

# Function to search for a movie using TMDB API
def search_movie(movie_title: str, year: str = "") -> Optional[Dict[str, Any]]:
    """Search for a movie using the TMDB API"""
    api_key = get_api_key()
    if not api_key:
        return None
    
    logging.info(f"Searching for '{movie_title}'{(' from ' + year) if year else ''}...")
    
    search_url = "https://api.themoviedb.org/3/search/movie"
    params = {
        "api_key": api_key,
        "query": movie_title,
        "language": "en-US",
        "include_adult": "false"
    }
    
    if year:
        params["year"] = year
    
    try:
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        search_results = response.json()
        
        if not search_results.get("results") or len(search_results["results"]) == 0:
            logging.warning(f"No movies found matching '{movie_title}'{' from ' + year if year else ''}.")
            return None
        
        # Get the first result
        movie = search_results["results"][0]
        logging.info(f"Found movie: {movie['title']} ({movie.get('release_date', 'Unknown')[:4]})")
        return movie
    
    except requests.exceptions.RequestException as e:
        logging.error(f"Error searching for movie: {str(e)}")
        return None

# Function to get detailed movie information
def get_movie_details(movie_id: int) -> Optional[Dict[str, Any]]:
    """Get detailed information about a movie using its ID"""
    api_key = get_api_key()
    if not api_key:
        return None
    
    details_url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {
        "api_key": api_key,
        "language": "en-US",
        "append_to_response": "keywords,credits"
    }
    
    try:
        response = requests.get(details_url, params=params)
        response.raise_for_status()
        logging.info(f"Successfully retrieved details for movie ID: {movie_id}")
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error getting movie details for ID {movie_id}: {str(e)}")
        return None

# Function to create a taste profile based on a movie
# Add these imports at the top of the file
import google.generativeai as genai
from dotenv import load_dotenv

# Add this function to check Gemini API status
def check_gemini_status() -> bool:
    """Check if Gemini API key is available and working"""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print(f"{Colors.RED}Error: GEMINI_API_KEY not found in environment variables.{Colors.ENDC}")
        return False
    
    try:
        # Configure the Gemini API
        genai.configure(api_key=api_key)
        # Simple test call
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content("Hello")
        return True
    except Exception as e:
        print(f"{Colors.RED}Error connecting to Gemini API: {str(e)}{Colors.ENDC}")
        return False

# Add this function to enhance tags using Gemini
def enhance_tags_with_gemini(taste_profile: Dict[str, Any]) -> List[str]:
    """Use Google's Gemini API to analyze movie details and generate more relevant tags"""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print(f"{Colors.RED}Error: GEMINI_API_KEY not found in environment variables.{Colors.ENDC}")
        return []
    
    try:
        # Extract data from taste_profile
        movie_title = taste_profile.get("movie", "")
        overview = taste_profile.get("overview", "")
        genres = taste_profile.get("genres", [])  # Fixed: access from taste_profile
        existing_keywords = taste_profile.get("keywords", [])
        directors = taste_profile.get("directors", [])
        cast = taste_profile.get("cast", [])
        
        if not overview:
            return []
        
        print(f"Analyzing movie themes with Gemini...")
        
        # Configure the Gemini API
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.0-pro') # Corrected model name
        
        # Prepare prompt for Gemini
        prompt = f"""
        Analyze this movie and extract 10-15 thematic tags focusing on:
        { "Genres and atmosphere" if '1' in taste_profile.get('preferences', []) else "" }
        { "Directing style and cinematography" if '2' in taste_profile.get('preferences', []) else "" }
        { "Story elements and plot structure" if '3' in taste_profile.get('preferences', []) else "" }
        
        Title: {movie_title}
        Director(s): {', '.join(directors)}
        Cast: {', '.join(cast)}
        Genres: {', '.join(genres)}
        
        Plot summary: {overview}
        
        Return a JSON object with two fields: 
        - "tags": an array of thematic tags (single words or short phrases)
        - "analysis": brief (50-word) explanation of thematic connections
        
        Example format:
        {{"tags": ["coming of age", "friendship", "loss", "redemption", "identity"], 
         "analysis": "A poignant exploration of adolescent identity through friendship and loss."}}
        """
        
        # Call Gemini API
        response = model.generate_content(prompt)
        response_text = response.text
        
        try:
            # Parse the JSON response
            # Imports moved to top level
            
            # Try to extract JSON from the response
            json_match = re.search(r'({.*})', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
                tags = result.get("tags", [])
                analysis = result.get("analysis", "")
                
                if analysis:
                    print(f"\n{Colors.GREEN}Theme Analysis:{Colors.ENDC} {analysis}")
                
                return tags
            else:
                # Fallback: try to extract comma-separated tags
                tags = [tag.strip() for tag in response_text.split(',') if tag.strip()]
                return tags
                
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract tags directly
            tags = [tag.strip() for tag in response_text.split(',') if tag.strip()]
            return tags
            
    except Exception as e:
        print(f"{Colors.RED}Error using Gemini for tag enhancement: {str(e)}{Colors.ENDC}")
        return []

# Add this function to get recommendations using Gemini
def get_gemini_recommendations(taste_profile: Dict[str, Any], all_recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Use Google's Gemini API to rank and personalize recommendations"""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key or not all_recommendations:
        return all_recommendations[:10]  # Return top 10 without ranking
    
    try:
        # Extract information from taste_profile
        movie_title = taste_profile.get("movie", "")
        overview = taste_profile.get("overview", "")
        genres = taste_profile.get("genres", [])  # Fixed: access from taste_profile
        keywords = taste_profile.get("keywords", [])
        enhanced_keywords = taste_profile.get("enhanced_keywords", [])
        
        # Prepare movie data for ranking
        movie_data = []
        for movie in all_recommendations[:15]:  # Limit to top 15 for Gemini processing
            movie_data.append({
                "id": movie["id"],
                "title": movie["title"],
                "overview": movie["overview"][:200] + "..." if len(movie["overview"]) > 200 else movie["overview"],
                "release_date": movie.get("release_date", "Unknown"),
                "vote_average": movie.get("vote_average", 0)
            })
        
        print(f"Using Gemini to personalize recommendations...")
        
        # Configure the Gemini API
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.0-pro') # Corrected model name
        
        # Enhanced prompt with preferences
        preference_context = []
        if '1' in taste_profile.get('preferences', []):
            preference_context.append("Prioritize similar genres and atmosphere")
        if '2' in taste_profile.get('preferences', []):
            preference_context.append("Focus on directorial style and cinematography")
        if '3' in taste_profile.get('preferences', []):
            preference_context.append("Emphasize story structure and thematic elements")
        
        # Prepare prompt for Gemini
        prompt = f"""
        Rank these movies based on similarity to the user's taste profile.
        
        User's original movie: {movie_title}
        Plot summary: {overview}
        Genres: {', '.join(genres)}
        Themes: {', '.join(keywords[:7] if keywords else [])}
        Enhanced themes: {', '.join(enhanced_keywords[:7] if enhanced_keywords else [])}
        
        User preferences: {', '.join(preference_context) if preference_context else "No specific preferences"}
        
        Here are the movies to rank (JSON array):
        {json.dumps(movie_data)}
        
        Return a JSON array with the movies ranked from most to least relevant to the user's taste.
        Each object should have 'id' and 'relevance_score' (0-100) fields.
        Example: [{{"id": 123, "relevance_score": 95}}, {{"id": 456, "relevance_score": 80}}]
        
        Only return the JSON array, no other text.
        """
        
        # Call Gemini API
        response = model.generate_content(prompt)
        response_text = response.text
        
        try:
            # Parse the JSON response
            # Imports moved to top level
            
            # Try to extract JSON from the response
            json_match = re.search(r'(\[.*\])', response_text, re.DOTALL)
            if json_match:
                ranked_movies = json.loads(json_match.group(1))
                
                # Validate response format
                if not all(["id" in m and "relevance_score" in m for m in ranked_movies]):
                    raise ValueError("Invalid response format from Gemini")
                
                # Create mapping of movie IDs to full data
                movie_map = {movie["id"]: movie for movie in all_recommendations}
                
                # Sort recommendations based on Gemini's ranking
                sorted_recommendations = []
                for ranked_movie in ranked_movies:
                    movie_id = ranked_movie["id"]
                    if movie_id in movie_map:
                        sorted_recommendations.append({
                            **movie_map[movie_id],
                            "relevance_score": ranked_movie["relevance_score"]
                        })
                
                # Filter out low relevance scores
                filtered_recommendations = [m for m in sorted_recommendations if m.get("relevance_score", 0) >= 60]
                
                # If filtering removed too many, use the original sorted list
                if len(filtered_recommendations) < 5 and sorted_recommendations:
                    return sorted_recommendations[:10]
                    
                return filtered_recommendations[:10]  # Return top 10 ranked results
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"{Colors.RED}Error parsing Gemini response: {str(e)}{Colors.ENDC}")
            return all_recommendations[:10]  # Fallback to original list
        
        return all_recommendations[:10]
    
    except Exception as e:
        print(f"{Colors.RED}Error in Gemini ranking: {str(e)}{Colors.ENDC}")
        return all_recommendations[:10]

def create_taste_profile(movie: Dict[str, Any], movie_details: Dict[str, Any], interactive: bool = True) -> Dict[str, Any]:
    """Create a taste profile based on movie details"""
    # Extract genres
    genres = [genre["name"] for genre in movie_details.get("genres", [])]
    
    # Extract keywords
    keywords = [keyword["name"] for keyword in movie_details.get("keywords", {}).get("keywords", [])]
    
    # Extract director information
    directors = []
    if movie_details.get("credits", {}).get("crew"):
        directors = [crew_member["name"] for crew_member in movie_details.get("credits", {}).get("crew", []) 
                    if crew_member.get("job") == "Director"]
    
    # Extract cast information (top 5 actors)
    cast = []
    if movie_details.get("credits", {}).get("cast"):
        cast = [cast_member["name"] for cast_member in movie_details.get("credits", {}).get("cast", [])[:5]]
    
    # Create taste profile
    taste_profile = {
        "movie": movie["title"],
        "genres": genres,
        "keywords": keywords,
        "directors": directors,
        "cast": cast,
        "overview": movie_details.get("overview", ""),
        "id": movie["id"],
        "release_date": movie.get("release_date", "Unknown"),
        "vote_average": movie_details.get("vote_average", "N/A")
    }
    
    # Display taste profile information (only in interactive mode)
    if interactive:
        print(f"Your Taste Profile:")
        print(f"Based on: {taste_profile['movie']} ({taste_profile['release_date'][:4] if taste_profile['release_date'] != 'Unknown' else 'Unknown'})")
        
        if genres:
            print(f"Genres: {', '.join(genres)}")
        
        if keywords:
            print(f"Key themes: {', '.join(keywords[:5])}")
        
        if directors:
            print(f"Director(s): {', '.join(directors)}")
        
        if cast:
            print(f"Main cast: {', '.join(cast)}")
        
        # Add user preference input only in interactive mode
        print(f"What did you love most about {movie['title']}?")
        print(f"1. The genre/vibe")
        print(f"2. The director's style")
        print(f"3. The story/plot")
        preference = input(f"Enter numbers (comma-separated): ").strip()
        
        taste_profile["preferences"] = [p.strip() for p in preference.split(',') if p.strip() in ['1', '2', '3']]
    else:
        # Default preferences when running in non-interactive mode
        taste_profile["preferences"] = []
    
    return taste_profile

def enhance_tags_with_ollama(taste_profile: Dict[str, Any]) -> List[str]:
    """Use Ollama to analyze movie details and generate more relevant tags"""
    if not check_ollama_status():
        # The check_ollama_status function now provides detailed error messages via logging
        # Just return the original keywords since Ollama is not available
        logging.info("Ollama not available, skipping tag enhancement.")
        return taste_profile.get("keywords", [])
    
    try:
        # Ollama is available, let's use it to enhance the tags
        logging.info("Attempting to enhance movie tags with Ollama...")
        
        # Get the existing keywords and other movie information
        keywords = taste_profile.get("keywords", [])
        movie_title = taste_profile.get("movie", "")
        overview = taste_profile.get("overview", "")
        genres = taste_profile.get("genres", [])
        
        # --- Ollama Integration Placeholder --- 
        # This section needs the actual Ollama API call logic.
        # For now, it falls back to existing keywords or generates basic ones.
        # Example prompt structure (needs implementation):
        # prompt = f"Analyze the movie '{movie_title}' with overview: '{overview}'. Genres: {', '.join(genres)}. Existing keywords: {', '.join(keywords)}. Generate 5-10 highly relevant and specific keywords or tags describing its core themes, style, and elements. Output only a comma-separated list of keywords."
        # ollama_response = requests.post("http://localhost:11434/api/generate", json={"model": "llama3", "prompt": prompt, "stream": False})
        # enhanced_keywords = ollama_response.json()['response'].split(',') # Simplified parsing
        # logging.info(f"Ollama generated tags: {enhanced_keywords}")
        # return [tag.strip() for tag in enhanced_keywords if tag.strip()]
        # --- End Placeholder ---

        # Current fallback logic:
        if keywords:
            logging.info(f"Found {len(keywords)} existing keywords. Using them as Ollama enhancement is not yet implemented.")
            return keywords
        else:
            # If no keywords were found, generate some based on the overview
            logging.info("No keywords found. Generating based on movie overview (basic method).")
            if overview:
                import re
                potential_keywords = re.findall(r'\b[A-Za-z]{4,}\b', overview)
                unique_keywords = list(set(potential_keywords))[:5]
                if unique_keywords:
                    logging.info(f"Generated {len(unique_keywords)} keywords from overview: {unique_keywords}")
                    return unique_keywords
            
            if genres:
                logging.info("Using genres as fallback keywords.")
                return genres
            
            logging.warning("Could not generate any keywords.")
            return []
            
    except Exception as e:
        logging.error(f"Error during Ollama tag enhancement placeholder: {str(e)}")
        # Fallback to original keywords
        return taste_profile.get("keywords", [])

# Function to get recommendations based on a movie
def get_movie_recommendations(movie_id: int) -> List[Dict[str, Any]]:
    """Get movie recommendations based on a movie ID"""
    api_key = get_api_key()
    if not api_key:
        return []
    
    rec_url = f"https://api.themoviedb.org/3/movie/{movie_id}/recommendations"
    params = {
        "api_key": api_key,
        "language": "en-US",
        "page": 1
    }
    
    try:
        response = requests.get(rec_url, params=params)
        response.raise_for_status()
        recommendations = response.json()
        
        results = recommendations.get("results", [])
        logging.info(f"Found {len(results)} direct recommendations for movie ID {movie_id}.")
        return results
    
    except requests.exceptions.RequestException as e:
        logging.error(f"Error getting recommendations for movie ID {movie_id}: {str(e)}")
        return []

# Function to get movies by genre
def get_movies_by_genre(genres: List[str]) -> List[Dict[str, Any]]:
    """Get movie recommendations based on genres"""
    api_key = get_api_key()
    if not api_key:
        return []
    if not genres:
        logging.warning("No genres provided for genre-based search.")
        return []

    # Get genre IDs
    genre_url = "https://api.themoviedb.org/3/genre/movie/list"
    params = {
        "api_key": api_key,
        "language": "en-US"
    }
    genre_ids = []
    try:
        response = requests.get(genre_url, params=params)
        response.raise_for_status()
        genre_list = response.json()
        
        if genre_list.get("genres"):
            genre_map = {g['name'].lower(): g['id'] for g in genre_list["genres"]}
            for genre_name in genres:
                genre_id = genre_map.get(genre_name.lower())
                if genre_id:
                    genre_ids.append(genre_id)
        
        if not genre_ids:
            logging.warning(f"Could not find IDs for genres: {genres}")
            return []
        logging.info(f"Found IDs for genres: {genres} -> {genre_ids}")

    except requests.exceptions.RequestException as e:
        logging.error(f"Error getting genre list: {str(e)}")
        return []
        
    # Get movies by genre
    discover_url = "https://api.themoviedb.org/3/discover/movie"
    params = {
        "api_key": api_key,
        "language": "en-US",
        "sort_by": "popularity.desc",
        "with_genres": ",".join(map(str, genre_ids)),
        "page": 1
    }
    
    try:
        response = requests.get(discover_url, params=params)
        response.raise_for_status()
        discovered_movies = response.json()
        results = discovered_movies.get("results", [])
        logging.info(f"Found {len(results)} movies for genres: {genres}")
        return results
    
    except requests.exceptions.RequestException as e:
        logging.error(f"Error getting movies by genre IDs {genre_ids}: {str(e)}")
        return []

# Function to get movies by keyword
def get_movies_by_keyword(keywords: List[str], limit: int = 3) -> List[Dict[str, Any]]:
    """Get movie recommendations based on keywords"""
    api_key = get_api_key()
    if not api_key or not keywords:
        if not keywords:
            logging.warning("No keywords provided for keyword-based search.")
        return []
    
    search_keywords = keywords[:limit]
    logging.info(f"Searching for movies based on keywords: {search_keywords}")
    all_results = []
    keyword_ids_found = {} # Cache found keyword IDs

    # Pre-fetch keyword IDs if possible (reduces redundant API calls)
    # This part is optional optimization

    for keyword in search_keywords:
        keyword_id = None
        try:
            # Search for the keyword ID
            if keyword.lower() in keyword_ids_found:
                keyword_id = keyword_ids_found[keyword.lower()]
            else:
                keyword_search_url = "https://api.themoviedb.org/3/search/keyword"
                params = {"api_key": api_key, "query": keyword, "page": 1}
                response = requests.get(keyword_search_url, params=params)
                response.raise_for_status()
                keyword_results = response.json()
                
                if keyword_results.get("results") and len(keyword_results["results"]) > 0:
                    keyword_id = keyword_results["results"][0]["id"]
                    keyword_ids_found[keyword.lower()] = keyword_id # Cache it
                    logging.info(f"Found keyword ID for '{keyword}': {keyword_id}")
                else:
                    logging.warning(f"Could not find keyword ID for '{keyword}'.")
                    continue # Skip this keyword if ID not found
            
            # Get movies with that keyword ID
            discover_url = "https://api.themoviedb.org/3/discover/movie"
            params = {
                "api_key": api_key,
                "language": "en-US",
                "sort_by": "popularity.desc",
                "with_keywords": str(keyword_id),
                "page": 1
            }
            
            response = requests.get(discover_url, params=params)
            response.raise_for_status()
            discovered_movies = response.json()
            
            keyword_movie_results = discovered_movies.get("results", [])
            if keyword_movie_results:
                logging.info(f"Found {len(keyword_movie_results)} movies for keyword '{keyword}' (ID: {keyword_id}). Adding top 5.")
                all_results.extend(keyword_movie_results[:5])
        
        except requests.exceptions.RequestException as e:
            logging.error(f"Error searching for keyword '{keyword}' or its movies: {str(e)}")
            continue
    
    # Remove duplicates based on movie ID
    unique_movies = {}
    for movie in all_results:
        if movie["id"] not in unique_movies:
            unique_movies[movie["id"]] = movie
    
    final_list = list(unique_movies.values())
    logging.info(f"Found {len(final_list)} unique movies across keywords: {search_keywords}")
    return final_list

# Function to get personalized recommendations (placeholder for Ollama ranking)
def get_ollama_recommendations(taste_profile: Dict[str, Any], all_recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rank and personalize recommendations. Placeholder for Ollama integration."""
    if not all_recommendations:
        logging.warning("No recommendations provided to personalize.")
        return []
    
    ollama_available = check_ollama_status() # Check status but don't rely on it yet
    
    logging.info("Personalizing recommendations based on taste profile (using basic scoring)...")
    
    try:
        # --- Ollama Ranking Placeholder ---
        # This section needs the actual Ollama API call for ranking.
        # Example prompt structure (needs implementation):
        # movie_list_str = "\n".join([f"- {m['title']} ({m.get('release_date', 'N/A')[:4]}): {m.get('overview', '')[:100]}..." for m in all_recommendations])
        # prompt = f"Based on this taste profile (liked movie: {taste_profile['movie']}, genres: {taste_profile['genres']}, keywords: {taste_profile.get('enhanced_keywords', taste_profile['keywords'])}, preferences: {taste_profile['preferences']}), rank the following movie recommendations for relevance:\n{movie_list_str}\nOutput only the list of movie titles in the best order."
        # ollama_response = requests.post("http://localhost:11434/api/generate", json={"model": "llama3", "prompt": prompt, "stream": False})
        # ranked_titles = ollama_response.json()['response'].split('\n') # Simplified parsing
        # ranked_movies = sorted(all_recommendations, key=lambda m: ranked_titles.index(m['title']) if m['title'] in ranked_titles else 999)
        # logging.info("Recommendations ranked using Ollama.")
        # return ranked_movies[:10]
        # --- End Placeholder ---

        # Current fallback: Simple scoring based on profile (without Ollama)
        scored_recommendations = []
        preferences = taste_profile.get('preferences', []) # Preferences from interactive mode, might be empty
        liked_genres = taste_profile.get('genres', [])
        liked_keywords = taste_profile.get('enhanced_keywords', taste_profile.get('keywords', []))
        # liked_directors = taste_profile.get('directors', []) # Not used in current scoring
        # liked_cast = taste_profile.get('cast', []) # Not used in current scoring

        # Fetch genre map once if needed for ID matching
        genre_map_by_id = {}
        if any(m.get('genre_ids') and not m.get('genres') for m in all_recommendations):
            api_key = get_api_key()
            if api_key:
                genre_url = "https://api.themoviedb.org/3/genre/movie/list"
                try:
                    response = requests.get(genre_url, params={"api_key": api_key, "language": "en-US"})
                    response.raise_for_status()
                    genre_list = response.json()
                    if genre_list.get("genres"):
                        genre_map_by_id = {g['id']: g['name'] for g in genre_list["genres"]}
                except requests.exceptions.RequestException as e:
                    logging.warning(f"Could not fetch genre list for ID mapping: {e}")

        for movie in all_recommendations:
            score = 0
            # Base score for being included
            score += 50 
            
            # Score based on vote average (scaled)
            score += min(15, (movie.get('vote_average', 0) or 0) * 1.5) # Max 15 points
            
            # Score based on popularity (log scaled, simple version)
            score += min(10, (movie.get('popularity', 0) or 0) ** 0.5 / 5) # Max 10 points

            # Check for genre matches (more points if preferred)
            movie_genres_names = []
            if movie.get('genres'): # Full genre info available
                 movie_genres_names = [genre['name'] for genre in movie.get('genres', [])]
            elif movie.get('genre_ids') and genre_map_by_id: # Only IDs available, use map
                movie_genres_names = [genre_map_by_id.get(gid) for gid in movie.get('genre_ids', []) if genre_map_by_id.get(gid)]
            
            genre_match_score = 0
            for liked_genre in liked_genres:
                if liked_genre in movie_genres_names:
                    genre_match_score += 3
                    if '1' in preferences: # User prefers genre/vibe (from interactive mode)
                        genre_match_score += 2
            score += min(20, genre_match_score) # Max 20 points for genres

            # Keyword matching (simple check, could be improved with embeddings)
            # This requires fetching keywords for each recommended movie - expensive!
            # Skipping keyword scoring for API efficiency for now.

            # Director/Cast matching (also requires fetching credits - expensive)
            # Skipping for now.

            scored_recommendations.append({
                **movie,
                'relevance_score': min(100, round(score)) # Cap at 100
            })
        
        # Sort by relevance score
        sorted_recommendations = sorted(scored_recommendations, key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        if ollama_available:
            logging.info("Using built-in scoring engine. Ollama ranking not implemented yet.")
        else:
            logging.info("Using built-in scoring engine (Ollama not available).")
        
        return sorted_recommendations[:10]
    
    except Exception as e:
        logging.error(f"Error in recommendation ranking: {str(e)}", exc_info=True) # Log traceback
        logging.warning("Falling back to basic recommendation sorting by popularity/rating.")
        # Fallback sorting
        return sorted(all_recommendations, 
                      key=lambda x: ((x.get('popularity', 0) or 0) + (x.get('vote_average', 0) or 0) * 10), 
                      reverse=True)[:10]

# Removed main() function and if __name__ == '__main__': block
# This script is now intended to be imported as a module by app.py
