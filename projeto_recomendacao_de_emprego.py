import numpy as np
import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed

# Configura o logging para registrar em arquivo
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

# Carrega e pré-processa os dados
def load_and_preprocess_data():
    try:
        logger.info("Carregando os datasets...")
        df_jobs = pd.read_csv('Combined_Jobs_Final.csv')
        df_interest = pd.read_csv('merge_de_interest_x_experience.csv')
    except FileNotFoundError as e:
        logger.error(f"Erro ao carregar arquivos: {e}")
        return None, None, None

    logger.info("Normalizando os salários...")
    scaler = StandardScaler()
    df_jobs['Salary'] = scaler.fit_transform(df_jobs[['Salary']])
    df_interest['Salary'] = scaler.fit_transform(df_interest[['Salary']])

    return df_jobs, df_interest, scaler

# Pré-processa o texto (transforma para minúsculas)
def preprocess_text(text):
    return text.lower()

# Configura TF-IDF e SVD
def setup_tfidf_vectorizer(df_jobs):
    logger.info("Configurando TF-IDF e SVD...")
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix_jobs = vectorizer.fit_transform(df_jobs['Job.Description'])
    svd = TruncatedSVD(n_components=50)
    return vectorizer, svd, svd.fit_transform(tfidf_matrix_jobs)

# Aplica k-NN para encontrar as vagas mais próximas
def apply_knn(tfidf_matrix_jobs_reduced):
    logger.info("Aplicando k-NN...")
    nn = NearestNeighbors(n_neighbors=5, metric='cosine')
    nn.fit(tfidf_matrix_jobs_reduced)
    return nn

# Processa um candidato e retorna recomendações
def process_candidate(index, candidate, df_jobs, nn, vectorizer, svd, scaler):
    logger.info(f"Procurando vagas para: {candidate['Position.Of.Interest']}")
    candidate_tfidf_reduced = svd.transform(vectorizer.transform([preprocess_text(candidate['Job.Description'])]))
    distances, indices = nn.kneighbors(candidate_tfidf_reduced)
    valid_indices = indices[0][indices[0] < len(df_jobs)]

    if valid_indices.size == 0:
        logger.warning(f"Todos os índices são inválidos para {candidate['Position.Of.Interest']}.")
        return pd.DataFrame()

    recommendations = df_jobs.iloc[valid_indices].copy()
    higher_salary_recommendations = recommendations[recommendations['Salary'] > candidate['Salary']]
    recommendations = higher_salary_recommendations if not higher_salary_recommendations.empty else recommendations
    recommendations['Salary'] = scaler.inverse_transform(recommendations[['Salary']])
    return recommendations

# Obtém recomendações para todos os candidatos
def get_recommendations(df_jobs, df_interest, nn, vectorizer, svd, scaler):
    logger.info("Obtendo recomendações...")
    return Parallel(n_jobs=-1)(delayed(process_candidate)(index, candidate, df_jobs, nn, vectorizer, svd, scaler)
                               for index, candidate in df_interest.iterrows())

# Calcula precisão, recall e f1-score
def calculate_metrics(y_true, y_pred):
    return (precision_score(y_true, y_pred, zero_division=1),
            recall_score(y_true, y_pred, zero_division=1),
            f1_score(y_true, y_pred, zero_division=1))

# Avalia o sistema com base nas recomendações
def evaluate_system(df_test, recommended_jobs, df_interest, nn, vectorizer, svd):
    logger.info("Avaliação do sistema...")
    y_true, y_pred, distances_list = [], [], []

    for i, candidate in df_interest.iterrows():
        if i < len(recommended_jobs) and not recommended_jobs[i].empty:
            candidate_tfidf_reduced = svd.transform(vectorizer.transform([preprocess_text(candidate['Job.Description'])]))
            distances, indices = nn.kneighbors(candidate_tfidf_reduced)

            for j, index in enumerate(indices[0]):
                if index < len(df_test):
                    y_true.append(1)  # Existe uma posição de interesse
                    y_pred.append(int(candidate['Position.Of.Interest'].lower() in df_test.iloc[index]['Position'].lower()))
                    distances_list.append(distances[0][j])
                else:
                    y_true.append(1)  # Posição válida, mas sem recomendação
                    y_pred.append(0)
        else:
            y_true.append(1)
            y_pred.append(0)

    precision, recall = calculate_metrics(y_true, y_pred)
    accuracy = np.mean(np.array(y_pred) == np.array(y_true))
    avg_distance = np.mean(distances_list) if distances_list else None

    if avg_distance:
        logger.info(f"Distância média das recomendações: {avg_distance:.4f}")
    else:
        logger.warning("Nenhuma distância disponível para calcular.")

    return precision, recall, accuracy, avg_distance

# Função principal
def main():
    df_jobs, df_interest, scaler = load_and_preprocess_data()
    df_train, df_test = train_test_split(df_jobs, test_size=0.2, random_state=42)
    vectorizer, svd, tfidf_matrix_train = setup_tfidf_vectorizer(df_train)
    nn = apply_knn(tfidf_matrix_train)

    unique_positions = df_interest['Position.Of.Interest'].unique()
    print("\nPosições disponíveis para recomendações:")
    for i, position in enumerate(unique_positions, start=1):
        print(f"{i}: {position}")

    while True:
        try:
            choice = int(input("\nDigite o número da posição desejada para procurar recomendações (ou 0 para sair): "))
            if choice == 0:
                print("Saindo do sistema.")
                break
            if 1 <= choice <= len(unique_positions):
                print(f"\nVocê selecionou: {unique_positions[choice - 1]}")
                break
            print(f"Por favor, digite um número entre 1 e {len(unique_positions)}.")
        except ValueError:
            print("Entrada inválida. Por favor, digite um número.")

    if choice != 0:
        recommended_jobs = get_recommendations(df_train, df_interest, nn, vectorizer, svd, scaler)
        precision, recall, accuracy, avg_distance = evaluate_system(df_test, recommended_jobs, df_interest, nn, vectorizer, svd)
        print(f"\nResultados da avaliação:\nPrecisão: {precision:.2f}\nRecall: {recall:.2f}\nAcurácia: {accuracy:.2f}")
        if avg_distance:
            print(f"Distância média das recomendações: {avg_distance:.4f}")

if __name__ == "__main__":
    main()
    
