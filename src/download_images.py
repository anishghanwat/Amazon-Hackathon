import os
import requests
from tqdm import tqdm

def download_image(url, save_path):
    """
    Download an image from a given URL and save it to the specified path.
    
    Args:
    url (str): The URL of the image to download.
    save_path (str): The path where the image will be saved.
    
    Returns:
    bool: True if download was successful, False otherwise.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return False

def download_images(url_list, save_directory):
    """
    Download multiple images from a list of URLs.
    
    Args:
    url_list (list): A list of image URLs to download.
    save_directory (str): The directory where images will be saved.
    
    Returns:
    int: The number of successfully downloaded images.
    """
    os.makedirs(save_directory, exist_ok=True)
    successful_downloads = 0
    
    for i, url in enumerate(tqdm(url_list, desc="Downloading images")):
        file_name = f"image_{i+1}.jpg"
        save_path = os.path.join(save_directory, file_name)
        
        if download_image(url, save_path):
            successful_downloads += 1
    
    return successful_downloads

if __name__ == "__main__":
    # Example usage
    image_urls = [
        #"https://m.media-amazon.com/images/I/71XfHPR36-L.jpg",
        "https://example.com/image1.jpg",
        "https://example.com/image2.jpg",
        # Add more URLs as needed
    ]
    save_dir = "downloaded_images"
    
    total_downloaded = download_images(image_urls, save_dir)
    print(f"Successfully downloaded {total_downloaded} images.")