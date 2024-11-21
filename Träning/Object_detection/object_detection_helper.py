from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np

def visualize_image_with_bounding_boxes_colored(image_path, bounding_boxes):
    """
    Visualizes an image with bounding boxes, each colored according to the rectanglelabels.
    Adds labels and colors to the legend.
    
    Parameters:
    - image_path: Path to the image file.
    - bounding_boxes: A list of bounding boxes, where each bounding box is represented as a dictionary
      with keys 'x', 'y', 'width', 'height', and optionally 'label'. Coordinates are normalized to [0, 1].
    """
    # Open the image file
    img = Image.open(image_path)
    fig, ax = plt.subplots(1)
    fig.set_size_inches(10, 10)
    ax.imshow(img)
    
    # Image dimensions
    img_width, img_height = img.size
    
    # Label to color mapping
    label_color_map = {
        'riktning_text': 'r',
        'another_label': 'g',
          # Example additional label
        # Add more labels and colors as needed
    }
    
    # Add bounding boxes
    for box in bounding_boxes:
        # Denormalize coordinates
        x = box['x'] * img_width //100
        y = box['y'] * img_height //100
        width = box['width'] * img_width //100
        height = box['height'] * img_height //100
        
        # Get color for the label
        label = box['rectanglelabels'][0]
        color = label_color_map.get(label, 'b')  # Default to blue if label not in map
        
        # Create a Rectangle patch
        rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor=color, facecolor='none', label=label)
        
        # Add the patch to the Axes
        ax.add_patch(rect)
    
    # Create legend from unique labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # Removing duplicates
    ax.legend(by_label.values(), by_label.keys())
    
    plt.show()

def visualize_image_with_bounding_boxes_colored_img(img, bounding_boxes):
    """
    Visualizes an image with bounding boxes, each colored according to the rectanglelabels.
    Adds labels and colors to the legend.
    
    Parameters:
    - image_path: Path to the image file.
    - bounding_boxes: A list of bounding boxes, where each bounding box is represented as a dictionary
      with keys 'x', 'y', 'width', 'height', and optionally 'label'. Coordinates are normalized to [0, 1].
    """
    # Open the image file
    fig, ax = plt.subplots(1)
    fig.set_size_inches(10, 10)
    ax.imshow(img.permute(1, 2, 0).numpy())
    
    # Image dimensions
    img_width, img_height = img.shape[0], img.shape[1]
    
    # Label to color mapping
    label_color_map = {
        'riktning_text': 'r',
        'another_label': 'g',  # Example additional label
        # Add more labels and colors as needed
    }
    print(bounding_boxes["scores"].cpu())
    if len(bounding_boxes["scores"].cpu()) != 0:
        print(np.argmax(bounding_boxes["scores"].cpu()))
        # Add bounding boxes
        for i in range(len(bounding_boxes["boxes"].cpu())):
            # Denormalize coordinates
            #if bounding_boxes["scores"].cpu()[i] < 0.1:
            #    continue
            box = bounding_boxes["boxes"].cpu()[i]

            print(box)
            x = box[0] 
            y = box[1]
            width = box[2]-box[0]
            height = box[3]-box[1]
            
            # Create a Rectangle patch
            rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor="red", facecolor='none', label=1)
            
            # Add the patch to the Axes
            ax.add_patch(rect)
    
    # Create legend from unique labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # Removing duplicates
    ax.legend(by_label.values(), by_label.keys())
    
    plt.show()

def visualize_image_with_bounding_boxes_colored_img_annotated(img, bounding_boxes, annotation):
    """
    Visualizes an image with bounding boxes, each colored according to the rectanglelabels.
    Adds labels and colors to the legend.
    
    Parameters:
    - image_path: Path to the image file.
    - bounding_boxes: A list of bounding boxes, where each bounding box is represented as a dictionary
      with keys 'x', 'y', 'width', 'height', and optionally 'label'. Coordinates are normalized to [0, 1].
    """
    # Open the image file
    fig, ax = plt.subplots(1)
    fig.set_size_inches(10, 10)
    ax.imshow(img.permute(1, 2, 0).numpy())
    
    # Image dimensions
    img_width, img_height = img.shape[0], img.shape[1]
    
    # Label to color mapping
    label_color_map = {
        'riktning_text': 'r',
        'another_label': 'g',  # Example additional label
        # Add more labels and colors as needed
    }
    print(bounding_boxes["scores"].cpu())
    if len(bounding_boxes["scores"].cpu()) != 0:
        print(np.argmax(bounding_boxes["scores"].cpu()))
        # Add bounding boxes
        for i in range(len(bounding_boxes["boxes"].cpu())):
            # Denormalize coordinates
            #if bounding_boxes["scores"].cpu()[i] < 0.1:
            #    continue
            box = bounding_boxes["boxes"].cpu()[i]

            print(box)
            x = box[0] 
            y = box[1]
            width = box[2]-box[0]
            height = box[3]-box[1]
            print("Prediction")
            print(x, y, width, height)
            # Create a Rectangle patch
            rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor="red", facecolor='none', label="Prediction")
            
            # Add the patch to the Axes
            ax.add_patch(rect)
      
      #draw annotation
    
        # Denormalize coordinates
        x = annotation['x'] * annotation["original_width"] //400
        y = annotation['y'] *  annotation["original_height"]//400
        width = annotation['width'] * annotation["original_width"] //400
        height = annotation['height'] * annotation["original_height"] //400
        
        # Get color for the label
        label = annotation['label'][0]
        color = label_color_map.get(label, 'b')  # Default to blue if label not in map
        
        print("Annotation")
        print(x, y, width, height)

        # Create a Rectangle patch
        rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor=color, facecolor='none', label="Annotation")
        
        # Add the patch to the Axes
        ax.add_patch(rect)
    
    # Create legend from unique labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # Removing duplicates
    ax.legend(by_label.values(), by_label.keys())
    
    plt.show()