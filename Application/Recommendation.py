def recommendation_movies(user_id):
    import pandas as pd
    import numpy as np

    # Bước 1: Đọc dữ liệu từ hai file CSV
    ratings_df = pd.read_csv('datasets/ratings.csv')  # Dữ liệu đánh giá phim của người dùng
    movies_df = pd.read_csv('datasets/movies.csv')    # Dữ liệu thông tin phim

    # Hiển thị một vài dòng đầu tiên để xem cấu trúc dữ liệu
    ratings_df.head(), movies_df.head()

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # Bước 2: Xử lý cột 'genres' (thể loại phim) thành văn bản có thể vector hóa
    # Thay ký tự '|' bằng khoảng trắng để mỗi thể loại là một từ
    movies_df['processed_genres'] = movies_df['genres'].str.replace('|', ' ', regex=False)

    # Bước 3: Vector hóa thể loại bằng TF-IDF
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(movies_df['processed_genres'])

    # Bước 4: Kết hợp bảng đánh giá và bảng phim theo movieId
    ratings_movies_df = pd.merge(ratings_df, movies_df, on='movieId')

    # Bước 5: Chọn một người dùng cụ thể để gợi ý (ví dụ: người dùng có userId = 1)

    user_ratings = ratings_movies_df[ratings_movies_df['userId'] == user_id]

    # Bước 6: Lọc các phim mà người dùng đánh giá từ 4.0 trở lên (xem như người dùng thích)
    liked_movies = user_ratings[user_ratings['rating'] >= 4.0]

    # Bước 7: Lấy chỉ số các phim mà người dùng này thích
    liked_movie_indices = movies_df[movies_df['movieId'].isin(liked_movies['movieId'])].index

    # Bước 8: Tính hồ sơ người dùng bằng trung bình TF-IDF của các phim đã thích
    user_profile = np.asarray(tfidf_matrix[liked_movie_indices].mean(axis=0))

    # Bước 9: Tính độ tương đồng cosine giữa hồ sơ người dùng và tất cả các phim
    similarities = cosine_similarity(user_profile, tfidf_matrix).flatten()

    # Bước 10: Gán điểm tương đồng vào bảng phim
    movies_df['similarity'] = similarities

    # Bước 11: Loại bỏ các phim mà người dùng đã xem để không gợi ý lại
    recommended_df = movies_df[~movies_df['movieId'].isin(liked_movies['movieId'])]

    # Bước 12: Sắp xếp và chọn ra 10 phim có độ tương đồng cao nhất
    top_recommendations = recommended_df.sort_values(by='similarity', ascending=False).head(10)

    # Bước 13: In ra danh sách 10 phim được gợi ý
    # return top_recommendations[['title', 'genres', 'similarity']]
    return top_recommendations[['title', 'genres']]
