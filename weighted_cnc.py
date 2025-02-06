import numpy as np
import json
import pickle
from neater_cnc_tableau import CncSimulator

class WeightedCNCObject:
    def __init__(self, cnc_object: CncSimulator, quasiprobability: float, negativity: float):
        """
        Represents a CNC object with an associated probability.
        
        Args:
            cnc_object (CncSimulator): The CNC object.
            probability (float): Probability associated with the CNC object.
        """
        probability = np.abs(quasiprobability)/negativity

        self.cnc_object = cnc_object
        self.negativity = negativity
        self.quasiprobability = quasiprobability
        self.probability = probability

    def to_dict(self):
        """Converts the object to a dictionary format for serialization."""
        return {
            "tableau": self.cnc_object.tableau.tolist(),
            "quasiprobability": self.quasiprobability,
            "negativity": self.negativity,
            "probability": self.probability,
            "n": self.cnc_object.n,
            "m": self.cnc_object.m
        }

    @classmethod
    def from_dict(cls, data):
        """Creates a WeightedCNCObject from a dictionary."""
        cnc_object = CncSimulator.from_tableau(
            data["n"], data["m"], np.array(data["tableau"], dtype=np.uint8)
        )
        return cls(cnc_object,data["quasiprobability"], data["negativity"])

    def save_to_file(self, filename: str):
        """Saves the object to a file in JSON format."""
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load_from_file(cls, filename: str):
        """Loads the object from a JSON file."""
        with open(filename, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


# Example: Sampling weighted objects
def sample_weighted_objects(objects: list[WeightedCNCObject], num_samples: int) -> list[WeightedCNCObject]:
    """
    Samples objects based on their probabilities.
    
    Args:
        objects (list[WeightedCNCObject]): List of weighted CNC objects.
        num_samples (int): Number of samples to draw.
    
    Returns:
        list[WeightedCNCObject]: List of sampled objects.
    """
    probabilities = [obj.probability for obj in objects]
    indices = np.random.choice(len(objects), size=num_samples, p=probabilities)
    return [objects[i] for i in indices]


# Example usage
if __name__ == "__main__":
    # Create a CNC object
    cnc_obj = CncSimulator(4, 2)
    
    # Create Weighted CNC objects
    weighted_obj1 = WeightedCNCObject(cnc_obj, 0.3)
    weighted_obj2 = WeightedCNCObject(cnc_obj, 0.7)
    objects = [weighted_obj1, weighted_obj2]
    
    # Save to a file
    weighted_obj1.save_to_file("weighted_obj1.json")

    # Load from a file
    loaded_obj = WeightedCNCObject.load_from_file("weighted_obj1.json")
    print("Loaded Object Probability:", loaded_obj.probability)

    # Sample based on probabilities
    sampled_objects = sample_weighted_objects(objects, num_samples=5)
    for obj in sampled_objects:
        print("Sampled Object Probability:", obj.probability)
