import json
import numpy as np
from neater_cnc_tableau import CncSimulator
from weighted_cnc import WeightedCNCObject

class WeightedCNCCollection:
    def __init__(self, objects: list[WeightedCNCObject]):
        """
        Represents a collection of weighted CNC objects forming a probability distribution.
        
        Args:
            objects (list[WeightedCNCObject]): List of weighted CNC objects.
        """
        # Ensure the probabilities sum to 1
        total_probability = sum(obj.probability for obj in objects)

        # ensure all negativity values are the same:
        negativity_set = set([obj.negativity for obj in objects])
        negativity_boolean = len(set([obj.negativity for obj in objects])) == 1
        
        # ensure that negativity is consistent
        total_negativity = sum(np.abs(obj.quasiprobability) for obj in objects)

        # check if probabilities sum to one:
        if not np.isclose(total_probability, 1.0):
            raise ValueError("The probabilities of all objects must sum to 1.")
        
        # check if all objects have same negativity:
        if not negativity_boolean:
            raise ValueError("All negativity values must be the same for a proper probability distribution.")
        
        # Check if manually computed negativity agrees with given negativity:
        elif not np.isclose(total_negativity, list(negativity_set)[0]):
            raise ValueError("Sum of absolute value of all quasiprobabilities should sum to one.")
        
        self.objects = objects

    def to_dict(self):
        """Converts the collection to a dictionary format for serialization."""
        return [obj.to_dict() for obj in self.objects]

    @classmethod
    def from_dict(cls, data):
        """Creates a WeightedCNCCollection from a dictionary."""
        objects = [WeightedCNCObject.from_dict(obj) for obj in data]
        return cls(objects)

    def save_to_file(self, filename: str):
        """Saves the collection to a file in JSON format."""
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load_from_file(cls, filename: str):
        """Loads the collection from a JSON file."""
        with open(filename, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def sample(self, num_samples: int) -> list[WeightedCNCObject]:
        """
        Samples objects based on their probabilities.
        
        Args:
            num_samples (int): Number of samples to draw.
        
        Returns:
            list[WeightedCNCObject]: List of sampled objects.
        """
        probabilities = [obj.probability for obj in self.objects]
        indices = np.random.choice(len(self.objects), size=num_samples, p=probabilities)
        return [self.objects[i] for i in indices]
