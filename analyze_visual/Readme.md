# Visual Analysis

Extraction and analysis of visual-based features from videos.

## Usage

**Main script:** analyze_visual.py

**Flags:**

* f: extract features from specific file<br/>
    E.g. python3 analyze_visual.py -f \<filename\>
* d: extract features from all files of a directory<br/>
     E.g. python3 analyze_visual.py -d \<directory_name\>   
* evaluate: analyze extracted features<br/>
     E.g. python3 analyze_visual.py -evaluate   
* debug: debug script<br/>
     E.g. python3 analyze_visual.py -debug

**Basic functions:**

* process_video :  extracts features from specific file
* dir_process_video : extracts features from all files of a directory
* analyze : analyzes extracted features
* script_debug : debug script


For farther information read the docstrings.

