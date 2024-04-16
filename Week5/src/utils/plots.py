import os
import matplotlib.pyplot as plt
CLASS_NAMES = [
        "brush_hair", "catch", "clap", "climb_stairs", "draw_sword", "drink", 
        "fall_floor", "flic_flac", "handstand", "hug", "kick", "kiss", "pick", 
        "pullup", "push", "ride_bike", "run", "shoot_ball", "shoot_gun", "situp", 
        "smoke", "stand", "sword", "talk", "turn", "wave", 
        "cartwheel", "chew", "climb", "dive", "dribble", "eat", "fencing", 
        "golf", "hit", "jump", "kick_ball", "laugh", "pour", "punch", "pushup", 
        "ride_horse", "shake_hands", "shoot_bow", "sit", "smile", "somersault", 
        "swing_baseball", "sword_exercise", "throw", "walk"
    ]

class Plots:
    @staticmethod
    def generate_per_class_accuracy_plot(per_class_accuracy, name):
        """
        Generates a bar plot of per-class accuracies.

        Args:
            per_class_accuracy (list): List of per-class accuracies.

        Returns:
            None
        """
        # Plot per-class accuracy
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(per_class_accuracy)), per_class_accuracy)
        plt.xlabel('Class')
        plt.ylabel('Accuracy')
        plt.title('Per-Class Accuracy')
        plt.xticks(range(len(per_class_accuracy)), [f"{CLASS_NAMES[i]}" for i in range(len(per_class_accuracy))], rotation=45, ha='right')
        plt.tight_layout()

        # Create directory if it doesn't exist
        output_dir = '/ghome/group01/mcv-c6-2024-team1/Week5/plots/'
        os.makedirs(output_dir, exist_ok=True)

        # Save the plot as an image
        plt.savefig(os.path.join(output_dir, f'{name}.png'))
