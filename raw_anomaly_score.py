# Step 4: Create a raw anomaly score for each frame of a video, given temporal summaries (Si)
import transformers
import torch
from transformers import pipeline


class Anomaly_Score:
    def __init__(self, pipe):
        self.pipe = pipe

    def augment(self, summarized_caption: str):
        return [
            {"role":"system", "content":(
                "If you were a law enforcement agency, rate the scene described on a scale "
                "from 0.0 (completely normal) to 1.0 (highly suspicious). "
                "Return ONLY a single float from {0.0,0.1,…,0.9,1.0}, nothing else."
            )},
            {"role":"user",   "content": summarized_caption},
        ]

    def batch_generate(self, summaries: list[str]) -> list[float]:
        # build list of message-lists
        batch_msgs = [self.augment(s) for s in summaries]
        # run the pipeline on the entire micro-batch
        outs = self.pipe(
            batch_msgs,
            max_new_tokens=8,
            pad_token_id=self.pipe.tokenizer.eos_token_id,
        )
        scores = []
        for out in outs:
            # pipeline returns list of lists if >1 input
            entry = out[0] if isinstance(out, list) else out
            # extract assistant reply
            gen = entry["generated_text"]
            # find the assistant’s message
            text = ""
            for msg in gen:
                if msg.get("role") == "assistant":
                    text = msg.get("content", "")
                    break
            # parse float
            try:
                scores.append(float(text.strip()))
            except:
                scores.append(-1.0)
        return scores


