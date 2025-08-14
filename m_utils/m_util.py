from transformers import BartTokenizer
import re

PAD = "<pad>" # 1
UNK = "<unk>" # 3
CLS = "<s>" # 0
SEP = "</s>" # 2
POS = "<pos>" # 50265
NEG = "<neg>" # 50266
KEYWORD = "<key>"

SPECIAL_TOKENS_MAP = {"additional_special_tokens": [POS, NEG]}


def get_tokenizer(config_dir):
    tokenizer = BartTokenizer.from_pretrained(config_dir)
    num_added_tokens = tokenizer.add_special_tokens(SPECIAL_TOKENS_MAP)
    special_token_id_dict = {
        "pad_token_id": tokenizer.pad_token_id,
    }
    return tokenizer, num_added_tokens, special_token_id_dict

def remove_spaces_around_special_tokens(text, special_tokens):
    for token in special_tokens:
        text = re.sub(rf'\s*{re.escape(token)}\s*', token, text)
    return text

def combine_tokens(output, tokenizer, mod = 'test_phrase'):
    return_sentence = []
    for batch in range(output.size(0)):
        out_tokens = tokenizer.decode(output[batch, :])
        # print(output[batch, :])
        # print(out_tokens)
        out_string = re.sub(r'(<\/?s>|<pad>)', r' \1 ', out_tokens)
        out_tokens = out_string.split()
        return_tokens = []
        for t in out_tokens:
            # print('t -----> ', t)
            if t == CLS:
                continue
            elif t == SEP or t == PAD:
                break
            elif t.upper() == 'NULL':
                return_tokens.append('NULL')
            else:
                # print("t --------> ", t)
                return_tokens.append(t)
            
        if mod == "test_phrase":
            return_sentence.append(remove_spaces_around_special_tokens(' '.join(return_tokens), special_tokens = ["<pos>", "<neg>"]))
        elif mod == "test_question":
            print(return_tokens)
            return_tokens = return_tokens[1:]
            return_sentence.append(' '.join(return_tokens))
        
    return return_sentence


