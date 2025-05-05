from flask import Flask, request, jsonify, render_template
import movie_recommender as mr
import os
from dotenv import load_dotenv

# Load environment variables including Gemini API key
load_dotenv()

app = Flask(__name__)

# Serve the main HTML page
@app.route('/')
def index():
    return render_template('index.html')

# API endpoint for getting recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    movie_title = data.get('movie')
    year = data.get('year', '') # Optional year
    preferences = data.get('preferences', [])  # User preferences from frontend

    if not movie_title:
        return jsonify({'error': 'Movie title is required'}), 400

    try:
        # --- Integration with movie_recommender logic --- 
        # 1. Search for the movie
        movie = mr.search_movie(movie_title, year)
        if not movie:
            return jsonify({'error': f"Movie '{movie_title}' not found"}), 404

        # 2. Get details
        movie_details = mr.get_movie_details(movie['id'])
        if not movie_details:
            return jsonify({'error': 'Could not retrieve movie details'}), 500

        # 3. Create taste profile with user preferences from request
        taste_profile = mr.create_taste_profile(movie, movie_details, interactive=False)
        
        # Add preferences from request
        taste_profile['preferences'] = preferences

        # 4. Enhance tags using Gemini instead of Ollama
        enhanced_tags = mr.enhance_tags_with_gemini(taste_profile)
        if enhanced_tags:
            taste_profile['enhanced_keywords'] = enhanced_tags

        # 5. Gather recommendations
        all_recommendations = []
        all_recommendations.extend(mr.get_movie_recommendations(movie['id']))
        all_recommendations.extend(mr.get_movies_by_genre(taste_profile['genres']))
        keywords_to_use = taste_profile.get('enhanced_keywords', taste_profile.get('keywords', []))
        all_recommendations.extend(mr.get_movies_by_keyword(keywords_to_use))

        # 6. Remove duplicates
        seen_ids = set()
        unique_recommendations = []
        for rec_movie in all_recommendations:
            if rec_movie['id'] not in seen_ids:
                seen_ids.add(rec_movie['id'])
                unique_recommendations.append(rec_movie)
        
        # 7. Get personalized/ranked recommendations using Gemini
        final_recommendations = mr.get_gemini_recommendations(taste_profile, unique_recommendations)

        # Format results for JSON
        results = [
            {
                'title': rec.get('title', 'N/A'),
                'year': rec.get('release_date', 'Unknown')[:4],
                'overview': rec.get('overview', ''),
                'rating': rec.get('vote_average', 'N/A'),
                'relevance_score': rec.get('relevance_score', 0),
                'poster_path': rec.get('poster_path', None)
            }
            for rec in final_recommendations
        ]

        return jsonify({
            'recommendations': results,
            'taste_profile': {
                'movie': taste_profile.get('movie'),
                'genres': taste_profile.get('genres', []),
                'keywords': taste_profile.get('keywords', [])[:5],
                'enhanced_keywords': taste_profile.get('enhanced_keywords', [])[:5],
                'directors': taste_profile.get('directors', []),
            }
        })

    except Exception as e:
        print(f"Error during recommendation: {e}") # Log error server-side
        return jsonify({'error': 'An internal error occurred while generating recommendations.'}), 500

if __name__ == '__main__':
    # Make sure TMDB_API_KEY is set
    if not mr.get_api_key():
        print("Exiting: TMDB_API_KEY not set in .env file.")
    else:
        # Use 0.0.0.0 to make it accessible on the network
        # Debug=True is helpful for development but should be False in production
        app.run(host='0.0.0.0', port=5000, debug=True)