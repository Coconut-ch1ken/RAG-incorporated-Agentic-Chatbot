import pandas as pd
import re
import os

# 1. Define the filenames (Make sure these match your actual files)
file1 = 'Cycle 1B First 50.csv'
file2 = 'Cycle 1B Rest.csv'

# 2. Check if files exist to avoid errors
if not os.path.exists(file1) or not os.path.exists(file2):
    print("Error: Input CSV files not found. Please ensure they are in the same folder as this script.")
else:
    # 3. Load the data
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df_combined = pd.concat([df1, df2], ignore_index=True)

    # 4. Define the parsing function
    def parse_full_description(text):
        data = {}
        if not isinstance(text, str):
            return {'Job ID': None, 'Work Term Duration': None, 'Location Type': None, 
                    'Job Summary': None, 'Job Responsibilities': None, 'Required Skills': None}

        # Extract Job ID (looking for 5-6 digit number in first few lines)
        lines = text.split('\n')
        job_id = None
        for line in lines[:20]:
            stripped = line.strip()
            if stripped.isdigit() and len(stripped) >= 5:
                job_id = stripped
                break
        data['Job ID'] = job_id

        # Helper to extract sections
        def extract_section(start_marker, end_markers):
            pattern = re.escape(start_marker) + r'\s*\n\s*\n([\s\S]*?)(?=\n\n(?:' + '|'.join(map(re.escape, end_markers)) + r')|$)'
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
            return None

        # Markers to look for
        end_markers = [
            "Job Responsibilities:", "Required Skills:", "Compensation and Benefits:",
            "Targeted Degrees and Disciplines:", "APPLICATION INFORMATION",
            "Transportation and Housing:", "Job Summary:", "Work Term Duration:",
            "Employment Location Arrangement:", "Special Job Requirements:"
        ]
        
        def get_end_markers(exclude):
            return [m for m in end_markers if m != exclude]

        # Extract fields
        data['Work Term Duration'] = extract_section("Work Term Duration:", get_end_markers("Work Term Duration:"))
        data['Location Type'] = extract_section("Employment Location Arrangement:", get_end_markers("Employment Location Arrangement:"))
        data['Job Summary'] = extract_section("Job Summary:", get_end_markers("Job Summary:"))
        data['Job Responsibilities'] = extract_section("Job Responsibilities:", get_end_markers("Job Responsibilities:"))
        data['Required Skills'] = extract_section("Required Skills:", get_end_markers("Required Skills:"))
        
        return data

    def extract_location_details(text):
        if not isinstance(text, str):
            return ""
        region_match = re.search(r'Region:\s*\n\s*\n(.*?)\n\n', text)
        city_match = re.search(r'Job - City:\s*\n\s*\n(.*?)\n\n', text)
        
        if city_match: return city_match.group(1).strip()
        if region_match: return region_match.group(1).strip()
        return ""

    # 5. Apply extraction
    print("Processing descriptions... this may take a moment.")
    parsed_data = df_combined['Full Description'].fillna('').apply(parse_full_description).apply(pd.Series)
    df_combined['Extracted_Location'] = df_combined['Full Description'].fillna('').apply(extract_location_details)

    # 6. Build final dataframe
    df_final = pd.DataFrame()
    df_final['Job ID'] = parsed_data['Job ID']
    df_final['Job Name'] = df_combined['Job Title']
    df_final['Company Name'] = df_combined['Organization']
    df_final['Work Term Duration'] = parsed_data['Work Term Duration']
    
    # Combine City and Type (e.g., "Toronto (Hybrid)")
    loc_type = parsed_data['Location Type'].fillna('Unknown')
    extracted_loc = df_combined['Extracted_Location'].fillna('')
    df_final['Location + Type'] = extracted_loc + " (" + loc_type + ")"
    
    df_final['Job Summary'] = parsed_data['Job Summary']
    df_final['Job Responsibilities'] = parsed_data['Job Responsibilities']
    df_final['Required Skills'] = parsed_data['Required Skills']

    # 7. Export
    output_filename = 'Processed_Jobs_Unified.xlsx'
    df_final.to_excel(output_filename, index=False)
    print(f"Success! File saved as: {output_filename}")