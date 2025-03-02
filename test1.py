import urllib.request as libreq
import re
import requests
import os
from xml.etree import ElementTree as ET
from time import sleep

def extract_arxiv_ids(xml_content):
    """
    Extract arXiv IDs from XML content containing multiple papers
    
    Args:
        xml_content (bytes): XML response from arXiv API
        
    Returns:
        list: List of arXiv IDs
    """
    # Decode bytes to string
    xml_str = xml_content.decode('utf-8')
    
    # Parse XML
    root = ET.fromstring(xml_str)
    
    # Initialize list to store IDs
    arxiv_ids = []
    
    # Find all entry elements
    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
        # Find ID element within entry
        id_elem = entry.find('{http://www.w3.org/2005/Atom}id')
        if id_elem is not None:
            # Extract ID from the URL
            full_id = id_elem.text
            # Use regex to extract just the arXiv ID portion
            match = re.search(r'/abs/(.+?)(?:v\d+)?$', full_id)
            if match:
                arxiv_ids.append(match.group(1))
    
    return arxiv_ids

def download_papers(arxiv_ids):
    """
    Download PDFs for given arXiv IDs
    
    Args:
        arxiv_ids (list): List of arXiv IDs to download
    """
    # Create a directory to store PDFs if it doesn't exist
    if not os.path.exists('papers'):
        os.makedirs('papers')
    
    # Download each paper
    for i, paper_id in enumerate(arxiv_ids, 1):
        pdf_url = f'https://arxiv.org/pdf/{paper_id}'
        try:
            print(f"\nDownloading paper {i} (ID: {paper_id})...")
            response = requests.get(pdf_url)
            
            if response.status_code == 200:
                # Save the PDF
                filename = f'papers/paper_{i}.pdf'
                with open(filename, 'wb') as f:
                    f.write(response.content)
                print(f"Successfully downloaded: {filename}")
                print(f"File size: {len(response.content):,} bytes")
            else:
                print(f"Failed to download paper {paper_id}. Status code: {response.status_code}")
                
            # Be nice to arXiv servers by waiting between downloads
            sleep(3)
            
        except Exception as e:
            print(f"Error downloading paper {paper_id}: {str(e)}")

# Main execution
def main():
    # Get the arXiv IDs
    url = 'http://export.arxiv.org/api/query?search_query=all:Do+Cell+Phones+Cause+Cancer&start=0&max_results=5'
    try:
        with libreq.urlopen(url) as response:
            xml_content = response.read()
            ids = extract_arxiv_ids(xml_content)
            
            print("Found ArXiv IDs:")
            for i, arxiv_id in enumerate(ids, 1):
                print(f"{i}. {arxiv_id}")
            
            print("\nStarting downloads...")
            download_papers(ids)
            
            print("\nDownload process completed!")
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

main()