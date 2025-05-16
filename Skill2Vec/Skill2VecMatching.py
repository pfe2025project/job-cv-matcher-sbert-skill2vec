from Skill2Vec.utils.convert_to_text import convert_to_text
from Skill2Vec.utils.extract_skills import extract_skills
from gensim.models import Word2Vec
from Skill2Vec.utils.skill2vec_matching import skillset_similarity
import os


class Skill2VecMatching:
    
    def __init__(self, model_path="Skill2Vec/models/skill2vec_10k_trained.model"):
        """
        Initialise le modèle Word2Vec pré-entrainé.
        
        Args:
        - model_path (str): Chemin vers le modèle Word2Vec pré-entrainé.
        """
        self.model = Word2Vec.load(model_path)
        
    def process_input(self, input_data):
        """
        Traite l'entrée (chemin de fichier ou texte brut).
        
        Args:
        - input_data (str or text): Chemin fichier ou texte brut.
        
        Returns:
        - str: Texte brut.
        """
        if isinstance(input_data, str) and os.path.isfile(input_data):
            return convert_to_text(input_data)
        return input_data
    
    def extract_skills_from_text(self, text):
        """
        Extrait les compétences depuis le texte.
        """
        skills = extract_skills(text)
        return self.process_skills(skills)
    
    def process_skills(self, skills):
        """
        Nettoie la liste de compétences (sans doublons, minuscules).
        """
        return list(set(skill["skill_name"].lower() for skill in skills))
    
    def calculate_similarity(self, cv_skills, job_skills):
        """
        Calcule la similarité entre deux listes de compétences.
        """
        return skillset_similarity(cv_skills, job_skills, self.model)
    
    def get_similarity_score(self, cv_input, job_input):
        """
        Calcule le score de similarité entre CV et offre donnés (chemin ou texte).
        
        Args:
        - cv_input (str or text): Chemin fichier ou texte brut du CV.
        - job_input (str or text): Chemin fichier ou texte brut de l'offre.
        
        Returns:
        - float: Score de similarité.
        """
        cv_text = self.process_input(cv_input)
        job_text = self.process_input(job_input)
        
        cv_skills = self.extract_skills_from_text(cv_text)
        job_skills = self.extract_skills_from_text(job_text)
        
        return self.calculate_similarity(cv_skills, job_skills)
