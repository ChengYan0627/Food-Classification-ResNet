from transformers import AutoImageProcessor, SiglipForImageClassification
import torch

class FoodClassifier:
    """Wrapper around SiglipForImageClassification for Food-101 dataset."""

    def __init__(self, device=None):
        """
        Initialize model and processor.

        Args:
            model_name (str): Hugging Face model ID.
            device (str, optional): 'cuda' or 'cpu'. Defaults to auto-detect.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "prithivMLmods/Food-101-93M"

        # Load model and processor
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = SiglipForImageClassification.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

        # Food-101 labels
        self.labels = {
            "0": "apple_pie", "1": "baby_back_ribs", "2": "baklava", "3": "beef_carpaccio", "4": "beef_tartare",
            "5": "beet_salad", "6": "beignets", "7": "bibimbap", "8": "bread_pudding", "9": "breakfast_burrito",
            "10": "bruschetta", "11": "caesar_salad", "12": "cannoli", "13": "caprese_salad", "14": "carrot_cake",
            "15": "ceviche", "16": "cheesecake", "17": "cheese_plate", "18": "chicken_curry", "19": "chicken_quesadilla",
            "20": "chicken_wings", "21": "chocolate_cake", "22": "chocolate_mousse", "23": "churros", "24": "clam_chowder",
            "25": "club_sandwich", "26": "crab_cakes", "27": "creme_brulee", "28": "croque_madame", "29": "cup_cakes",
            "30": "deviled_eggs", "31": "donuts", "32": "dumplings", "33": "edamame", "34": "eggs_benedict",
            "35": "escargots", "36": "falafel", "37": "filet_mignon", "38": "fish_and_chips", "39": "foie_gras",
            "40": "french_fries", "41": "french_onion_soup", "42": "french_toast", "43": "fried_calamari", "44": "fried_rice",
            "45": "frozen_yogurt", "46": "garlic_bread", "47": "gnocchi", "48": "greek_salad", "49": "grilled_cheese_sandwich",
            "50": "grilled_salmon", "51": "guacamole", "52": "gyoza", "53": "hamburger", "54": "hot_and_sour_soup",
            "55": "hot_dog", "56": "huevos_rancheros", "57": "hummus", "58": "ice_cream", "59": "lasagna",
            "60": "lobster_bisque", "61": "lobster_roll_sandwich", "62": "macaroni_and_cheese", "63": "macarons", "64": "miso_soup",
            "65": "mussels", "66": "nachos", "67": "omelette", "68": "onion_rings", "69": "oysters",
            "70": "pad_thai", "71": "paella", "72": "pancakes", "73": "panna_cotta", "74": "peking_duck",
            "75": "pho", "76": "pizza", "77": "pork_chop", "78": "poutine", "79": "prime_rib",
            "80": "pulled_pork_sandwich", "81": "ramen", "82": "ravioli", "83": "red_velvet_cake", "84": "risotto",
            "85": "samosa", "86": "sashimi", "87": "scallops", "88": "seaweed_salad", "89": "shrimp_and_grits",
            "90": "spaghetti_bolognese", "91": "spaghetti_carbonara", "92": "spring_rolls", "93": "steak", "94": "strawberry_shortcake",
            "95": "sushi", "96": "tacos", "97": "takoyaki", "98": "tiramisu", "99": "tuna_tartare", "100": "waffles"
        }

    def predict(self, image):
        """
        Run inference on a single PIL image.
    
        Args:
            image (PIL.Image.Image): Preprocessed image to classify.
    
        Returns:
            dict: {'label': str, 'confidence': float}
        """
        # Preprocess image
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
    
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze()
    
        # Prediction
        top_idx = torch.argmax(probs).item()
        label = self.labels[str(top_idx)]
        confidence = probs[top_idx].item()
    
        return {"label": label, "confidence": confidence}

