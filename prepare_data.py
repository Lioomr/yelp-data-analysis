import os

# Old file names
old_business = "yelp_academic_dataset_business.json"
old_review = "yelp_academic_dataset_review.json"

# New file names
new_business = "business.json"
new_review = "review.json"
small_review = "review_small.json"

# 🔹 Step 1: Rename files
if os.path.exists(old_business):
    os.rename(old_business, new_business)
    print("Business file renamed")

if os.path.exists(old_review):
    os.rename(old_review, new_review)
    print("Review file renamed")

# 🔹 Step 2: Reduce review size
with open(new_review, "r", encoding="utf-8") as f_in, open(small_review, "w", encoding="utf-8") as f_out:
    for i, line in enumerate(f_in):
        if i < 100000:   # keep first 100k rows
            f_out.write(line)
        else:
            break

print("Small review file created: review_small.json")