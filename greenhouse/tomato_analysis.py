# tomato_analysis.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
from datetime import datetime


class TomatoAnalyzer:
    def __init__(self):
        self.models_loaded = False
        self.load_ml_models()

    def load_ml_models(self):
        """Load ML models (if available) or create stubs"""
        try:
            self.model_biomass = joblib.load('biomass_model.pkl')
            self.model_height = joblib.load('height_model.pkl')
            self.model_leaf_area = joblib.load('leaf_area_model.pkl')
            self.models_loaded = True
            print("✅ ML models loaded")
        except:
            print("⚠️  ML models not found, using synthetic predictions")
            self.models_loaded = False

    def analyze_tomato_image(self, image_path):
        """Main tomato image analysis"""
        print(f"🔍 Analyzing image: {image_path}")

        # Load and preprocess image
        original_image = cv2.imread(image_path)
        if original_image is None:
            print("❌ Error loading image")
            return None

        # Convert to RGB
        rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        height, width = rgb_image.shape[:2]

        # 1. Green area segmentation
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

        # Green color range for tomatoes
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([95, 255, 255])
        green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

        # Morphological operations to improve mask
        kernel = np.ones((5, 5), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

        # 2. Calculate morphological features
        features = self.calculate_morphological_features(green_mask, width, height)

        # 3. ML predictions
        predictions = self.make_predictions(features)

        # 4. Visualize results
        self.visualize_results(original_image, rgb_image, green_mask, features, predictions, image_path)

        return features, predictions

    def calculate_morphological_features(self, mask, img_width, img_height):
        """Calculate plant morphological features"""
        # Green pixels area
        green_pixels = np.sum(mask > 0)
        total_pixels = img_width * img_height
        green_ratio = green_pixels / total_pixels

        # Find contours for size measurements
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Take the largest contour
            main_contour = max(contours, key=cv2.contourArea)

            # Plant height and width
            x, y, w, h = cv2.boundingRect(main_contour)
            plant_height = h
            plant_width = w

            # Approximate leaf count through contour analysis
            leaf_count = self.estimate_leaf_count(contours)

            # Leaf area (in pixels)
            leaf_area = cv2.contourArea(main_contour)
        else:
            plant_height = plant_width = leaf_count = leaf_area = 0

        features = {
            'green_pixel_ratio': green_ratio,
            'plant_height_pixels': plant_height,
            'plant_width_pixels': plant_width,
            'leaf_count_estimate': leaf_count,
            'leaf_area_pixels': leaf_area,
            'image_width': img_width,
            'image_height': img_height
        }

        print("📊 Calculated features:")
        for key, value in features.items():
            print(f"   {key}: {value:.4f}")

        return features

    def estimate_leaf_count(self, contours):
        """Estimate leaf count through contour analysis"""
        if not contours:
            return 0

        # Filter small contours (noise)
        min_contour_area = 100
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

        # Count significant contours as leaf estimate
        return min(len(filtered_contours), 20)  # Limit to reasonable maximum

    def make_predictions(self, features):
        """Predict growth parameters"""
        # Prepare features for ML models
        feature_vector = [
            features['green_pixel_ratio'],
            features['plant_height_pixels'],
            features['plant_width_pixels'],
            features['leaf_count_estimate'],
            24.5,  # temperature (example)
            65.0,  # humidity (example)
            450.0,  # CO2 (example)
            8000.0  # light intensity (example)
        ]

        if self.models_loaded:
            # Real model predictions
            biomass = self.model_biomass.predict([feature_vector])[0]
            height = self.model_height.predict([feature_vector])[0]
            leaf_area = self.model_leaf_area.predict([feature_vector])[0]
        else:
            # Synthetic predictions based on features
            biomass = features['green_pixel_ratio'] * 100 + features['plant_height_pixels'] * 0.1
            height = features['plant_height_pixels'] * 0.25  # example calibration pixels -> cm
            leaf_area = features['leaf_area_pixels'] * 0.02  # example calibration

        predictions = {
            'biomass_g': max(0, biomass),
            'height_cm': max(0, height),
            'leaf_area_cm2': max(0, leaf_area),
            'growth_stage': self.determine_growth_stage(biomass, height)
        }

        print("🤖 ML predictions:")
        print(f"   Biomass: {predictions['biomass_g']:.1f} g")
        print(f"   Height: {predictions['height_cm']:.1f} cm")
        print(f"   Leaf area: {predictions['leaf_area_cm2']:.1f} cm²")
        print(f"   Growth stage: {predictions['growth_stage']}")

        return predictions

    def determine_growth_stage(self, biomass, height):
        """Determine plant growth stage"""
        if biomass < 20:
            return "Seedling"
        elif biomass < 50:
            return "Vegetative"
        elif biomass < 100:
            return "Flowering"
        else:
            return "Fruiting"

    def visualize_results(self, original, rgb, mask, features, predictions, image_path):
        """Visualize all analysis stages"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Tomato Plant Analysis using Computer Vision', fontsize=16, fontweight='bold')

        # 1. Original image
        axes[0, 0].imshow(rgb)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')

        # 2. HSV representation
        hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
        axes[0, 1].imshow(hsv)
        axes[0, 1].set_title('HSV Color Space')
        axes[0, 1].axis('off')

        # 3. Green areas mask
        axes[0, 2].imshow(mask, cmap='gray')
        axes[0, 2].set_title(f'Green Segmentation ({features["green_pixel_ratio"] * 100:.1f}%)')
        axes[0, 2].axis('off')

        # 4. Plant contours
        contour_image = rgb.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_image, contours, -1, (255, 0, 0), 2)
        axes[1, 0].imshow(contour_image)
        axes[1, 0].set_title('Plant Contours')
        axes[1, 0].axis('off')

        # 5. Bounding box
        bbox_image = rgb.copy()
        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(main_contour)
            cv2.rectangle(bbox_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(bbox_image, f'H: {h}px', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        axes[1, 1].imshow(bbox_image)
        axes[1, 1].set_title('Plant Dimensions')
        axes[1, 1].axis('off')

        # 6. Prediction results
        axes[1, 2].axis('off')
        results_text = f"""ANALYSIS RESULTS:

Biomass: {predictions['biomass_g']:.1f} g
Height: {predictions['height_cm']:.1f} cm
Leaf Area: {predictions['leaf_area_cm2']:.1f} cm²
Growth Stage: {predictions['growth_stage']}

MORPHOLOGICAL FEATURES:
Green Pixels: {features['green_pixel_ratio'] * 100:.1f}%
Height (px): {features['plant_height_pixels']}
Width (px): {features['plant_width_pixels']}
Leaves: {features['leaf_count_estimate']} pcs"""

        axes[1, 2].text(0.1, 0.9, results_text, transform=axes[1, 2].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()

        # Save results
        filename = os.path.basename(image_path).split('.')[0]
        output_path = f"tomato_analysis_{filename}_{datetime.now().strftime('%H%M%S')}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Results saved: {output_path}")

        plt.show()


# Usage instructions
def main():
    analyzer = TomatoAnalyzer()

    # Specify path to your tomato image
    image_path = "test.jpg"  # REPLACE WITH YOUR PATH

    if os.path.exists(image_path):
        features, predictions = analyzer.analyze_tomato_image(image_path)

        # Generate report for paper
        print("\n" + "=" * 50)
        print("📑 PAPER REPORT:")
        print("=" * 50)
        print(f"Image: {os.path.basename(image_path)}")
        print(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nMorphological parameters:")
        print(f"- Green pixel ratio: {features['green_pixel_ratio'] * 100:.2f}%")
        print(f"- Plant height: {features['plant_height_pixels']} px")
        print(f"- Canopy width: {features['plant_width_pixels']} px")
        print(f"- Leaf count: {features['leaf_count_estimate']}")
        print(f"- Leaf area: {features['leaf_area_pixels']:.0f} px²")

        print("\nML predictions:")
        print(f"- Biomass: {predictions['biomass_g']:.1f} g/plant")
        print(f"- Height: {predictions['height_cm']:.1f} cm")
        print(f"- Leaf area: {predictions['leaf_area_cm2']:.1f} cm²")
        print(f"- Growth stage: {predictions['growth_stage']}")

    else:
        print(f"❌ File {image_path} not found!")
        print("📸 Please place tomato photo in program folder and specify correct filename")


if __name__ == "__main__":
    main()