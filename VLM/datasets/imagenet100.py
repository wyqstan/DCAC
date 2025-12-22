import os
import torchvision
from collections import OrderedDict

imagenet100_classes = ['stingray', 'ostrich', 'jay', 'American dipper', 'spotted salamander', 'alligator lizard', 'Komodo dragon', 'wolf spider', 'african grey parrot', 'jacamar', 'red-breasted merganser', 'tusker', 'jellyfish', 'brain coral', 'snail', 'white stork', 'dowitcher', 'albatross', 'Beagle', 'Otterhound', 'Lakeland Terrier', 'Giant Schnauzer', 'Cocker Spaniel', 'Australian Kelpie', 'Miniature Pinscher', 'Samoyed', 'Cardigan Welsh Corgi', 'Standard Poodle', 'Egyptian Mau', 'snow leopard', 'jaguar', 'polar bear', 'cockroach', 'hare', 'orangutan', 'gibbon', 'guenon', 'black-and-white colobus', "Geoffroy's spider monkey", 'bath towel', 'bell tower', 'birdhouse', 'bookstore', 'cardboard box / carton', 'chainsaw', 'chiffonier', 'cornet', 'cradle', 'crate', 'Crock Pot', 'desktop computer', 'rotary dial telephone', 'dog sled', 'electric locomotive', 'flagpole', 'four-poster bed', 'French horn', 'frying pan', 'fur coat', 'gas pump', 'gong', 'greenhouse', 'jeep', 'ladle', 'lighter', 'one-piece bathing suit', 'marimba', 'ocarina', 'overskirt', 'palace', 'paper towel', 'railroad car', 'pencil sharpener', 'Pickelhaube', 'pier', 'piggy bank', 'pool table', 'power drill', 'race car', 'radio', 'rifle', 'sarong', 'schooner', 'sewing machine', 'sliding door', 'sunglasses', 'swim trunks / shorts', 'syringe', 'front curtain', 'tow truck', 'trimaran', 'wardrobe', 'water tower', 'shipwreck', 'crossword', 'ice cream', 'cabbage', 'promontory', 'baseball player', 'hen of the woods mushroom']
# imagenet_templates = ["itap of a {}.",
#                         "a bad photo of the {}.",
#                         "a origami {}.",
#                         "a photo of the large {}.",
#                         "a {} in a video game.",
#                         "art of the {}.",
#                         "a photo of the small {}."]
imagenet_templates = ["a photo of a {}."]
class ImageNet100():

    dataset_dir = 'imagenet100'

    def __init__(self, root, preprocess):

        self.dataset_dir = root
        self.image_dir = root

        test_preprocess = preprocess
        
        self.test = torchvision.datasets.ImageFolder(self.image_dir, transform=test_preprocess)
        
        self.template = imagenet_templates
        self.classnames = imagenet100_classes
    
    def read_classnames(text_file):
        """Return a dictionary containing
        key-value pairs of <folder name>: <class name>.
        """
        classnames = OrderedDict()
        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                classnames[folder] = classname
        return classnames
