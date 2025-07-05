import os
import pandas as pd
from datetime import datetime


data_commented_path = "."

comments_files = [
            os.path.join(data_commented_path, file)
            for file in os.listdir(data_commented_path)
            if os.path.isfile(os.path.join(data_commented_path, file)) and file.lower().endswith(".csv")
        ]

df_merged = pd.concat((pd.read_csv(f) for f in comments_files), ignore_index=True)
merge_path = os.path.join(data_commented_path, f"merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
df_merged.to_csv(merge_path, index=False)

