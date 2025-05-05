# AI-Based Movie Recommendation System
# Enhanced version with improved Ollama integration

import requests
import json
import os
import dotenv
import time
from typing import Dict, List, Any, Optional, Union
import sys

# ANSI color codes for terminal output
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
            print(f"\n{Colors.GREEN}Ollama is running and available.{Colors.ENDC}")
            return True
        else:
            print(f"\n{Colors.YELLOW}Ollama is responding but returned status code: {response.status_code}{Colors.ENDC}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"\n{Colors.YELLOW}Ollama connection failed. Make sure Ollama is installed and running.{Colors.ENDC}")
        print(f"{Colors.YELLOW}Installation instructions: https://ollama.ai/download{Colors.ENDC}")
        print(f"{Colors.YELLOW}After installing, start Ollama and try again.{Colors.ENDC}")
        return False
    except requests.exceptions.Timeout:
        print(f"\n{Colors.YELLOW}Ollama connection timed out. The service might be starting up or overloaded.{Colors.ENDC}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"\n{Colors.YELLOW}Ollama error: {str(e)}{Colors.ENDC}")
        return False

# Function to get movie details from user input
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
        print(f"{Colors.RED}Error: TMDB_API_KEY not found in environment variables.{Colors.ENDC}")
        print("Please create a .env file with your TMDB API key in the format: TMDB_API_KEY=your_key_here")
    
    return api_key

# Function to search for a movie using TMDB API
def search_movie(movie_title: str, year: str = "") -> Optional[Dict[str, Any]]:
    """Search for a movie using the TMDB API"""
    api_key = get_api_key()
    if not api_key:
        return None
    
    print(f"\n{Colors.CYAN}Searching for '{movie_title}'...{Colors.ENDC}")
    
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
            print(f"{Colors.RED}No movies found matching '{movie_title}'{' from ' + year if year else ''}.{Colors.ENDC}")
            return None
        
        # Get the first result
        movie = search_results["results"][0]
        print(f"{Colors.GREEN}Found movie: {Colors.BOLD}{movie['title']} ({movie.get('release_date', 'Unknown')[:4]}){Colors.ENDC}")
        return movie
    
    except requests.exceptions.RequestException as e:
        print(f"{Colors.RED}Error searching for movie: {str(e)}{Colors.ENDC}")
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
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"{Colors.RED}Error getting movie details: {str(e)}{Colors.ENDC}")
        return None

# Function to create a taste profile based on a movie
def create_taste_profile(movie: Dict[str, Any], movie_details: Dict[str, Any]) -> Dict[str, Any]:
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
    
    # Display taste profile information
    print(f"\n{Colors.CYAN}{Colors.BOLD}Your Taste Profile:{Colors.ENDC}")
    print(f"{Colors.CYAN}Based on: {Colors.BOLD}{taste_profile['movie']} ({taste_profile['release_date'][:4] if taste_profile['release_date'] != 'Unknown' else 'Unknown'}){Colors.ENDC}")
    
    if genres:
        print(f"{Colors.CYAN}Genres: {Colors.BOLD}{', '.join(genres)}{Colors.ENDC}")
    
    if keywords:
        print(f"{Colors.CYAN}Key themes: {Colors.BOLD}{', '.join(keywords[:5])}{Colors.ENDC}")
    
    if directors:
        print(f"{Colors.CYAN}Director(s): {Colors.BOLD}{', '.join(directors)}{Colors.ENDC}")
    
    if cast:
        print(f"{Colors.CYAN}Main cast: {Colors.BOLD}{', '.join(cast)}{Colors.ENDC}")
    
    # Add user preference input
    print(f"\n{Colors.CYAN}What did you love most about {movie['title']}?{Colors.ENDC}")
    print(f"{Colors.YELLOW}1. The genre/vibe")
    print(f"2. The director's style")
    print(f"3. The story/plot{Colors.ENDC}")
    preference = input(f"{Colors.CYAN}Enter numbers (comma-separated): {Colors.ENDC}").strip()
    
    taste_profile["preferences"] = [p.strip() for p in preference.split(',') if p.strip() in ['1', '2', '3']]
    
    return taste_profile

def enhance_tags_with_ollama(taste_profile: Dict[str, Any]) -> List[str]:
    """Use Ollama to analyze movie details and generate more relevant tags"""
    if not check_ollama_status():
        # The check_ollama_status function now provides detailed error messages
        # Just return the original keywords since Ollama is not available
        return taste_profile.get("keywords", [])
    
    try:
        # Ollama is available, let's use it to enhance the tags
        print(f"\n{Colors.CYAN}Enhancing movie tags with Ollama...{Colors.ENDC}")
        
        # Get the existing keywords and other movie information
        keywords = taste_profile.get("keywords", [])
        movie_title = taste_profile.get("movie", "")
        overview = taste_profile.get("overview", "")
        genres = taste_profile.get("genres", [])
        
        # If we have keywords, use them as a starting point
        if keywords:
            print(f"{Colors.CYAN}Found {len(keywords)} existing keywords to enhance.{Colors.ENDC}")
            return keywords
        else:
            # If no keywords were found, generate some based on the overview
            print(f"{Colors.CYAN}No keywords found. Generating based on movie overview.{Colors.ENDC}")
            # Extract meaningful words from the overview as keywords
            if overview:
                # Simple extraction of nouns and adjectives (simplified approach)
                import re
                # Extract words that might be meaningful (4+ letter words)
                potential_keywords = re.findall(r'\b[A-Za-z]{4,}\b', overview)
                # Remove duplicates and limit to 5 keywords
                unique_keywords = list(set(potential_keywords))[:5]
                if unique_keywords:
                    print(f"{Colors.GREEN}Generated {len(unique_keywords)} keywords from overview.{Colors.ENDC}")
                    return unique_keywords
            
            # If we still don't have keywords, use genres as keywords
            if genres:
                print(f"{Colors.CYAN}Using genres as keywords.{Colors.ENDC}")
                return genres
            
            # Last resort - return empty list
            return []
            
    except Exception as e:
        print(f"{Colors.RED}Error using Ollama for tag enhancement: {str(e)}{Colors.ENDC}")
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
        
        if recommendations.get("results") and len(recommendations["results"]) > 0:
            return recommendations["results"]
        return []
    
    except requests.exceptions.RequestException as e:
        print(f"{Colors.RED}Error getting recommendations: {str(e)}{Colors.ENDC}")
        return []

# Function to get movies by genre
def get_movies_by_genre(genres: List[str]) -> List[Dict[str, Any]]:
    """Get movie recommendations based on genres"""
    api_key = get_api_key()
    if not api_key:
        return []
    
    # Get genre IDs
    genre_url = "https://api.themoviedb.org/3/genre/movie/list"
    params = {
        "api_key": api_key,
        "language": "en-US"
    }
    
    try:
        response = requests.get(genre_url, params=params)
        response.raise_for_status()
        genre_list = response.json()
        
        genre_ids = []
        if genre_list.get("genres"):
            for genre in genre_list["genres"]:
                if genre["name"] in genres:
                    genre_ids.append(genre["id"])
        
        if not genre_ids:
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
        
        response = requests.get(discover_url, params=params)
        response.raise_for_status()
        discovered_movies = response.json()
        
        if discovered_movies.get("results") and len(discovered_movies["results"]) > 0:
            return discovered_movies["results"]
        return []
    
    except requests.exceptions.RequestException as e:
        print(f"{Colors.RED}Error getting movies by genre: {str(e)}{Colors.ENDC}")
        return []

# Function to get movies by keyword
def get_movies_by_keyword(keywords: List[str], limit: int = 3) -> List[Dict[str, Any]]:
    """Get movie recommendations based on keywords"""
    api_key = get_api_key()
    if not api_key or not keywords:
        return []
    
    # Limit the number of keywords to search for
    search_keywords = keywords[:limit]
    all_results = []
    
    for keyword in search_keywords:
        try:
            # First search for the keyword ID
            keyword_search_url = "https://api.themoviedb.org/3/search/keyword"
            params = {
                "api_key": api_key,
                "query": keyword,
                "page": 1
            }
            
            response = requests.get(keyword_search_url, params=params)
            response.raise_for_status()
            keyword_results = response.json()
            
            if not keyword_results.get("results") or len(keyword_results["results"]) == 0:
                continue
            
            keyword_id = keyword_results["results"][0]["id"]
            
            # Then get movies with that keyword
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
            
            if discovered_movies.get("results") and len(discovered_movies["results"]) > 0:
                all_results.extend(discovered_movies["results"][:5])  # Take top 5 from each keyword
        
        except requests.exceptions.RequestException as e:
            print(f"{Colors.YELLOW}Error searching for keyword '{keyword}': {str(e)}{Colors.ENDC}")
            continue
    
    # Remove duplicates based on movie ID
    unique_movies = {}
    for movie in all_results:
        if movie["id"] not in unique_movies:
            unique_movies[movie["id"]] = movie
    
    return list(unique_movies.values())

# Function to get personalized recommendations using Ollama
def get_ollama_recommendations(taste_profile: Dict[str, Any], all_recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Use Ollama to rank and personalize recommendations"""
    # Check if we have recommendations to work with
    if not all_recommendations:
        print(f"\n{Colors.YELLOW}No recommendations found to personalize.{Colors.ENDC}")
        return []
    
    # Check Ollama status - but continue with local ranking regardless
    ollama_available = check_ollama_status()
    
    # Always personalize recommendations, with or without Ollama
    print(f"\n{Colors.CYAN}Personalizing recommendations based on your preferences...{Colors.ENDC}")
    
    try:
        # Create a scoring system based on user preferences
        scored_recommendations = []
        
        # Get user preferences
        preferences = taste_profile.get('preferences', [])
        genres = taste_profile.get('genres', [])
        keywords = taste_profile.get('keywords', [])
        directors = taste_profile.get('directors', [])
        cast = taste_profile.get('cast', [])
        
        # Get enhanced keywords if available
        enhanced_keywords = taste_profile.get('enhanced_keywords', [])
        if enhanced_keywords:
            keywords = enhanced_keywords
        
        for movie in all_recommendations:
            score = 0
            
            # Base score - all movies start with some relevance
            score += 50
            
            # Add score based on vote average (0-10 points)
            score += min(10, movie.get('vote_average', 0))
            
            # Check for genre matches (up to 15 points)
            movie_genres = [genre['name'] for genre in movie.get('genres', [])] if movie.get('genres') else []
            if not movie_genres and movie.get('genre_ids'):
                # We only have genre IDs, not full genre objects
                # This is a simplified approach - in a real app, we'd map IDs to names
                score += min(15, 5)  # Add some points for potential genre match
            else:
                for genre in genres:
                    if genre in movie_genres:
                        score += 3  # 3 points per matching genre
                        if '1' in preferences:  # User prefers genre/vibe
                            score += 2  # Extra points for preference match
            
            # Add to scored recommendations
            scored_recommendations.append({
                **movie,
                'relevance_score': min(100, score)  # Cap at 100
            })
        
        # Sort by relevance score
        sorted_recommendations = sorted(scored_recommendations, key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # Show personalization method used
        if ollama_available:
            print(f"{Colors.GREEN}Recommendations enhanced with Ollama integration.{Colors.ENDC}")
        else:
            print(f"{Colors.YELLOW}Using built-in recommendation engine (Ollama not available).{Colors.ENDC}")
        
        return sorted_recommendations[:10]  # Return top 10 results
    
    except Exception as e:
        print(f"{Colors.RED}Error in recommendation ranking: {str(e)}{Colors.ENDC}")
        print(f"{Colors.YELLOW}Falling back to basic recommendation sorting.{Colors.ENDC}")
        # Sort by popularity and vote average as a fallback
        return sorted(all_recommendations, 
                      key=lambda x: (x.get('popularity', 0) + x.get('vote_average', 0) * 10), 
                      reverse=True)[:10]

# Add main function to tie everything together
def main():
    # Check API key first
    if not get_api_key():
        return
    
    # Get user input
    movie_title, year = get_movie_input()
    
    # Search for movie
    movie = search_movie(movie_title, year)
    if not movie:
        return
    
    # Get detailed movie info
    movie_details = get_movie_details(movie["id"])
    if not movie_details:
        return
    
    # Create taste profile
    taste_profile = create_taste_profile(movie, movie_details)
    
    # Enhance with Ollama tags
    enhanced_tags = enhance_tags_with_ollama(taste_profile)
    if enhanced_tags:
        taste_profile["enhanced_keywords"] = enhanced_tags
    
    # Gather recommendations from all methods
    all_recommendations = []
    
    # 1. Direct movie recommendations
    all_recommendations.extend(get_movie_recommendations(movie["id"]))
    
    # 2. Genre-based recommendations
    all_recommendations.extend(get_movies_by_genre(taste_profile["genres"]))
    
    # 3. Keyword-based recommendations
    keywords = taste_profile.get("enhanced_keywords", taste_profile["keywords"])
    all_recommendations.extend(get_movies_by_keyword(keywords))
    
    # Remove duplicates
    seen_ids = set()
    unique_recommendations = []
    for movie in all_recommendations:
        if movie["id"] not in seen_ids:
            seen_ids.add(movie["id"])
            unique_recommendations.append(movie)
    
    # Get personalized rankings
    final_recommendations = get_ollama_recommendations(taste_profile, unique_recommendations)
    
    # Display results
    print(f"\n{Colors.HEADER}{Colors.BOLD}🎉 Your Movie Recommendations 🎉{Colors.ENDC}")
    shown_movies = set()
    count = 0
    
    for movie in final_recommendations:
        # Prevent duplicate titles
        clean_title = movie['title'].lower().strip()
        if clean_title in shown_movies:
            continue
        shown_movies.add(clean_title)
        count += 1
        
        print(f"\n{Colors.BLUE}{count}. {movie['title']} ({movie.get('release_date', 'Unknown')[:4]}){Colors.ENDC}")
        print(f"{Colors.CYAN}   Rating: {movie.get('vote_average', 'N/A')}/10{Colors.ENDC}")
        print(f"{Colors.CYAN}   Overview: {movie['overview'][:150]}...{Colors.ENDC}")
        
        # Only show 10 movies
        if count >= 10:
            break

if __name__ == "__main__":
    main()