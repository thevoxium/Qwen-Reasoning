
import requests
from bs4 import BeautifulSoup
import json
import time
from typing import Optional, List, Dict
import re

class AtCoderDataCollector:
    def __init__(self, username: str, password: str):
        self.session = requests.Session()
        self.base_url = "https://atcoder.jp"
        self.username = username
        self.password = password
        self.logged_in = False
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })
    
    def login(self) -> bool:
        try:
            login_url = f"{self.base_url}/login"
            print("Fetching login page...")
            response = self.session.get(login_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            form = soup.find('form')
            if not form:
                print("Error: Could not find login form")
                return False
                
            csrf_token = None
            csrf_input = form.find('input', {'name': 'csrf_token'})
            if csrf_input:
                csrf_token = csrf_input.get('value')
            
            if not csrf_token:
                print("Error: Could not find CSRF token")
                return False
            
            print("Found CSRF token...")
            
            login_data = {
                'csrf_token': csrf_token,
                'username': self.username,
                'password': self.password
            }
            
            print("Submitting login form...")
            response = self.session.post(
                login_url,
                data=login_data,
                allow_redirects=True
            )
            response.raise_for_status()
            
            print("Checking login status...")
            verification_page = self.session.get(f"{self.base_url}/home")
            
            soup = BeautifulSoup(verification_page.text, 'html.parser')
            
            if self.username.lower() in verification_page.text.lower():
                print("Login successful!")
                self.logged_in = True
                return True
            else:
                print("Login seems to have failed.")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"Network error during login: {str(e)}")
            return False
        except Exception as e:
            print(f"Unexpected error during login: {str(e)}")
            return False


    def get_problem_data(self, contest_id: str, task_id: str) -> Optional[Dict]:
        if not self.logged_in:
            return None
            
        url = f"{self.base_url}/contests/{contest_id}/tasks/{task_id}"
        print(f"Fetching problem from: {url}")
        response = self.session.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        task_statement = soup.find('div', id='task-statement')
        if not task_statement:
            print(f"Could not find task statement div for {task_id}")
            return None
            
        # Find the English version of the problem
        lang_div = task_statement.find('span', class_='lang-en')
        if not lang_div:
            print(f"Could not find English version for {task_id}")
            return None
        
        # Extract title from the page header
        title = soup.find('span', class_='h2')
        if not title:
            print(f"Could not find title for {task_id}")
            return None
        
        title_text = title.text.strip()
        
        problem_data = {
            'title': title_text,
            'description': '',
            'constraints': '',
            'input_format': '',
            'output_format': '',
            'sample_tests': []
        }
        
        # Each major section is in a div with class='part'
        parts = lang_div.find_all('div', class_='part')
        for part in parts:
            section = part.find('section')
            if not section:
                continue
                
            header = section.find('h3')
            if not header:
                continue
                
            header_text = header.text.lower()
            
            # For problem statement, we need to get everything in the section after h3
            if 'problem statement' in header_text:
                # Get all p tags in this section
                description_parts = []
                for p in section.find_all('p'):
                    description_parts.append(p.text.strip())
                # Join all parts with newlines
                problem_data['description'] = '\n'.join(description_parts)
                
            else:
                # For other sections, continue with the previous logic
                content_div = section.find('div')
                if not content_div:
                    continue
                    
                content = content_div.text.strip()
                
                if 'constraints' in header_text:
                    problem_data['constraints'] = content
                elif 'input' in header_text and 'format' in header_text:
                    problem_data['input_format'] = content
                elif 'output' in header_text and 'format' in header_text:
                    problem_data['output_format'] = content
                    
            # Handle sample tests separately
            if 'sample input' in header_text:
                # Look for the next section which should be sample output
                next_section = part.find_next_sibling('div', class_='part')
                if next_section and 'sample output' in next_section.find('h3').text.lower():
                    input_pre = section.find('pre')
                    output_pre = next_section.find('section').find('pre')
                    if input_pre and output_pre:
                        problem_data['sample_tests'].append({
                            'input': input_pre.text.strip(),
                            'output': output_pre.text.strip()
                        })
        
        return problem_data


    def get_first_ac_solution(self, contest_id: str, task_id: str) -> Optional[str]:
        if not self.logged_in:
            return None
            
        url = f"https://atcoder.jp/contests/{contest_id}/submissions?f.Task={task_id}&f.LanguageName=C%2B%2B&f.Status=AC&f.User="
        print(f"Fetching submissions from: {url}")
        response = self.session.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        # Find the first accepted submission row - it will have class 'table-success'
        submission_rows = soup.find_all('tr')
        print(len(submission_rows))
        submission_id = None
        for row in submission_rows:
            # Skip header row
            if row.find('th'):
                continue
            submission_link = row.find_all('td')[-1].find('a')
            if submission_link:
                href = submission_link.get('href', '')
                if href:
                    submission_id = href.split('/')[-1]
                    break
        
        if not submission_id:
            print(f"No accepted submission found for {task_id}")
            return None
        
        # Get the submission details page
        solution_url = f"{self.base_url}/contests/{contest_id}/submissions/{submission_id}"
        print(f"Fetching solution from: {solution_url}")
        response = self.session.get(solution_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the code in the submission-code pre tag
        code = soup.find('pre', id='submission-code')
        if not code:
            print(f"Could not find code for submission {submission_id}")
            return None
            
        return code.text.strip()


    def collect_arc_data(self, start_contest: int, end_contest: int) -> List[Dict]:
        output_file = "atcoder_training_data.json"
        
        # Try to load existing data
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
                print(f"Loaded {len(dataset)} existing problems from {output_file}")
        except (FileNotFoundError, json.JSONDecodeError):
            dataset = []
            print("Starting fresh data collection")
        
        for contest_num in range(start_contest, end_contest + 1):
            contest_id = f"abc{contest_num}"
            print(f"\nProcessing contest {contest_id}...")
            
            for task_letter in ['a', 'b', 'c', 'd']:
                task_id = f"{contest_id}_{task_letter}"
                print(f"\nProcessing task {task_id}...")
                
                # Check if we already have this task
                if any(item.get('title', '').startswith(f"{task_letter.upper()} -") and 
                      f"abc{contest_num}" in item.get('title', '').lower() 
                      for item in dataset):
                    print(f"Task {task_id} already exists in dataset, skipping...")
                    continue
                
                try:
                    problem_data = self.get_problem_data(contest_id, task_id)
                    if not problem_data:
                        print(f"Could not fetch problem data for {task_id}")
                        continue
                        
                    solution = self.get_first_ac_solution(contest_id, task_id)
                    if not solution:
                        print(f"Could not fetch solution for {task_id}")
                        continue
                        
                    problem_data['solution'] = solution
                    dataset.append(problem_data)
                    
                    # Save after each successful task
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(dataset, f, ensure_ascii=False, indent=2)
                    print(f"Saved progress to {output_file} ({len(dataset)} problems)")
                    
                    time.sleep(2)
                    
                except Exception as e:
                    print(f"Error processing {task_id}: {str(e)}")
                    # Save progress even if there's an error
                    if dataset:
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(dataset, f, ensure_ascii=False, indent=2)
                        print(f"Saved progress after error to {output_file} ({len(dataset)} problems)")
                    time.sleep(5)  # Longer delay after error
                    continue
        
        return dataset

# Update the main block to use the new changes
if __name__ == "__main__":
    USERNAME = "your_atcoder_username"
    PASSWORD = "your_atcoder_password"
    START_CONTEST = 210
    END_CONTEST = 300
    
    print("Initializing AtCoder data collection...")
    collector = AtCoderDataCollector(USERNAME, PASSWORD)
    
    if not collector.login():
        print("Login failed. Please check your credentials and try again.")
        exit(1)
    
    print("Starting data collection...")
    dataset = collector.collect_arc_data(START_CONTEST, END_CONTEST)
    
    # Final save is handled in collect_abc_data
    print(f"Collection complete. Total problems collected: {len(dataset)}")
