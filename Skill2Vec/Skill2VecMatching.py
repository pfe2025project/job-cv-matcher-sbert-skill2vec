from utils.convert_to_text import convert_to_text
from utils.extract_skills import extract_skills
from gensim.models import Word2Vec
from utils.skill2vec_matching import skillset_similarity

class Skill2VecMatching:
    
    def __init__(self, cv_input, job_input, model_path="models/skill2vec_10k_trained.model"):
        """
        Initialise le module de compétences avec le chemin du CV, de l'offre d'emploi, 
        et le modèle Word2Vec pré-entrainé.
        
        Args:
        - cv_input (str or text): Le chemin du fichier CV ou le texte brut du CV.
        - job_input (str or text): Le chemin du fichier de l'offre d'emploi ou le texte brut de l'offre.
        - model_path (str): Le chemin du modèle Word2Vec pré-entrainé.
        """
        self.cv_text = self.process_input(cv_input)
        self.job_text = self.process_input(job_input)
        self.model = Word2Vec.load(model_path)
        
        self.cv_skills = self.extract_skills_from_text(self.cv_text)
        self.job_skills = self.extract_skills_from_text(self.job_text)
        
        
    def process_input(self, input_data):
        """
        Traite l'entrée du CV ou de l'offre (chemin de fichier ou texte brut).
        
        Args:
        - input_data (str or text): Le chemin du fichier ou texte brut.
        
        Returns:
        - str: Le texte du fichier ou de l'entrée brute.
        """
        if isinstance(input_data, str):  # Vérifie si c'est un chemin de fichier
            return convert_to_text(input_data)
        return input_data  # Si c'est déjà du texte brut, le retourner tel quel
    
    def extract_skills_from_text(self, text):
        """
        Extrait les compétences du texte passé en argument.
        
        Args:
        - text (str): Le texte à partir duquel extraire les compétences.
        
        Returns:
        - list: Liste des compétences extraites du texte.
        """
        skills = extract_skills(text)
        return self.process_skills(skills)
    
    def process_skills(self, skills):
        """
        Traite les compétences extraites (enlève les doublons et les met en minuscules).
        
        Args:
        - skills (list): Liste des compétences extraites.
        
        Returns:
        - list: Liste des compétences sans doublons et en minuscules.
        """
        return list(set(skill["skill_name"].lower() for skill in skills))
    
    def calculate_similarity(self):
        """
        Calcule la similarité entre les compétences du CV et celles de l'offre d'emploi.
        
        Returns:
        - float: Score de similarité entre les compétences du CV et de l'offre.
        """
        return skillset_similarity(self.cv_skills, self.job_skills, self.model)
    
    def display_top_skills(self):
        """
        Affiche les 10 premières compétences de l'offre d'emploi extraites.
        """
        for skill in self.job_skills[:10]:
            print(f"Skill: {skill['skill_name']}, Type: {skill['skill_type']}, Match Type: {skill['match_type']}, Score: {skill['score']}")
            
    def get_similarity_score(self):
        """
        Retourne le score de similarité calculé entre les compétences du CV et de l'offre.
        
        Returns:
        - float: Le score de similarité.
        """
        return self.calculate_similarity()
