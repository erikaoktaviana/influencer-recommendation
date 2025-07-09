import pandas as pd
from flask import Flask, request, jsonify
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Flask API!"})


# Load data 
df = pd.read_csv('Influencers.csv')

# delete duplicate data
df.drop_duplicates(inplace=True)

# delete rows that are entirely empty
df.dropna(how='all', inplace=True)

# Cleaning Data
def convert_followers(x):
    if isinstance(x, str):
        x = x.replace(',','').strip()
        if 'M' in x:
            return float(x.replace('M', '')) * 1_000_000
        elif 'K' in x:
            return float(x.replace('K', '')) * 1_000
        else:
            return float(x)
    return x

def convert_percentage(x):
    if isinstance(x, str):
        x = x.strip().replace('%', '')
        try:
            return float(x)
        except:
            return None
    return x


# Apply cleaning to columns
df['Followers'] = df['Followers'].apply(convert_followers)
df['Avg. Likes'] = df['Avg. Likes'].apply(convert_followers)
df['Avg.Comment'] = df['Avg.Comment'].apply(convert_followers)
df['Eng. Rate'] = df['Eng. Rate'].apply(convert_percentage)
df['Growth Rate'] = df['Growth Rate'].apply(convert_percentage)

# handle empty values
df['Followers'] = df['Followers'].fillna(df['Followers'].median())
df['Avg. Likes'] = df['Avg. Likes'].fillna(df['Avg. Likes'].median())
df['Avg.Comment'] = df['Avg.Comment'].fillna(df['Avg.Comment'].median())
df['Eng. Rate'] = df['Eng. Rate'].fillna(df['Eng. Rate'].median())
df['Growth Rate'] = df['Growth Rate'].fillna(df['Growth Rate'].median())
df['Profile'] = df['Profile'].fillna('https://www.instagram.com/default_profile')  


@app.route('/get_influencers', methods=['GET'])
def get_influencers():

    # take 10 unique categories
    selected_categories = df['Category'].drop_duplicates().head(10).tolist()
    filtered_df = df[df['Category'].isin(selected_categories)]

    # Group influencer berdasarkan category
    grouped_data = {}
    for category in selected_categories:
        category_influencers = filtered_df[filtered_df['Category'] == category][['Username', 'Category', 'Followers','Profile']].to_dict(orient='records')
        grouped_data[category] = category_influencers
        

    return jsonify(grouped_data)



@app.route('/recommend', methods=['POST'])
def recommend():
    input_data = request.get_json()

    # input validation: 'Username' and ‘Category’ are in the request body
    if not input_data or 'Username' not in input_data or 'Category' not in input_data:
        return jsonify({"error": "'Username' dan 'Category' wajib ada dalam body request"}), 400
    
    # store input data for processing
    input_influencer = input_data['Username']
    input_category = input_data['Category']

    # filter the DataFrame based on the selected category
    filtered_df = df[df['Category'].str.lower() == input_category.lower()]

    # check if the influencer is in the category
    if input_influencer not in filtered_df['Username'].values:
        return jsonify({"error": "Influencer tidak ditemukan dalam kategori yang dipilih"}), 400

    # features used for similarity calculation
    features = ['Followers', 'Avg. Likes', 'Avg.Comment', 'Eng. Rate', 'Growth Rate']

    #  Fill the blank value in the ‘Growth Rate’ column with 0 and ensure there are no negative values
    filtered_df.loc[:, 'Growth Rate'] = filtered_df['Growth Rate'].fillna(0).apply(lambda x: max(x, 0))

    # normalization using min max scaler
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(filtered_df[features])

    # calculate the cosine similarity value
    similarity = cosine_similarity(normalized_features)

    # create a DataFrame to store the similarity values between influencers
    similarity_df = pd.DataFrame(similarity, index=filtered_df['Username'], columns=filtered_df['Username'])
    influencer_similarity = similarity_df.loc[input_influencer].drop(index=input_influencer)

    # take the top 5 influencers
    top_recommendations = influencer_similarity.sort_values(ascending=False).head(5)

    # take followers and profile data
    profile_data = df.set_index('Username')['Profile'].to_dict()
    followers_data = df.set_index('Username')['Followers'].to_dict()

    
    # recommendation result format
    recommendations_with_profiles = {
        influencer: {
            "similarity_score": score,
            "profile_url": profile_data.get(influencer, 'https://www.instagram.com/default_profile'),
             "followers": followers_data.get(influencer, 0)
        }
        for influencer, score in top_recommendations.items()
    }

    # Returns the recommendation result in JSON format
    return jsonify(recommendations_with_profiles)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)