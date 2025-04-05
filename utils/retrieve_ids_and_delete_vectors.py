from dotenv import load_dotenv
import os
import re
from pinecone import Pinecone
from pinecone.data import Index

load_dotenv(override=True)

api_key = os.getenv("PINECONE_API_KEY")

pinecone = Pinecone(api_key=api_key)

index_name = "arxiv-rag-metadata"

def get_ids(index, prefixes):
    all_matching_ids = []
    for prefix in prefixes:
        list_response = index.list(prefix=prefix)
        all_matching_ids.extend(
            [item for sublist in list_response for item in (sublist if isinstance(sublist, list) else [sublist])]
        )

    filtered_ids = [
        id_ for id_ in all_matching_ids 
        if isinstance(id_, str) and re.match(r"^\d{4}\.\d{4,5}$", id_)
    ]
    excluded_ids = [id_ for id_ in all_matching_ids if id_ not in filtered_ids]

    return filtered_ids, excluded_ids

def export_ids(filtered_ids, excluded_ids, filtered_file="filtered_ids.txt", unfiltered_file="unfiltered_ids.txt"):
    with open(filtered_file, "w") as f:
        f.write("\n".join(filtered_ids))
    print(f"Filtered IDs saved to {filtered_file}")

    with open(unfiltered_file, "w") as f:
        f.write("\n".join(filtered_ids) + "\n")
        f.write("\n".join(excluded_ids))
    print(f"Unfiltered IDs saved to {unfiltered_file}")

def delete_in_batches(index, ids, batch_size=1000):
    for i in range(0, len(ids), batch_size):
        batch = ids[i:i + batch_size]
        response = index.delete(ids=batch)
        print(f"Deleted batch {i // batch_size + 1}: {len(batch)} IDs")
    print("All batches processed.")

index_description = pinecone.describe_index(index_name)
index_host = index_description['host']
index = Index(api_key=api_key, host=index_host)

prefixes = ["2411"]
filtered_ids, excluded_ids = get_ids(index, prefixes)

export_ids(filtered_ids, excluded_ids)

# delete_in_batches(index, filtered_ids)