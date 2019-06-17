class NERUtil():
    def __init__(self):
        pass

    def load_ner_data(self, path):
        data = []
        data_label = []
        with open(path) as fr:
            lines = fr.readlines()
        sent_, tag_ = [], []
        for line in lines:
            if line != '\n':
                [char, label] = line.strip().split()
                sent_.append(char)
                tag_.append(label)
            else:
                data.append(' '.join(sent_))
                #data_label.append(' '.join(tag_))
                data_label.append(tag_)
                sent_, tag_ = [], []
        return data, data_label

    def _get_chunk_type(self, tok, idx_to_tag):
        """
        Args:
            tok: id of token, ex 4
            idx_to_tag: dictionary {4: "B-PER", ...}
        Returns:
            tuple: "B", "PER"
        """
        tag_name = idx_to_tag[tok]
        tag_class = tag_name.split('-')[0]
        tag_type = tag_name.split('-')[-1]
        return tag_class, tag_type

    def get_chunks(self, seq, tags):
        """Given a sequence of tags, group entities and their position
        Args:
            seq: [4, 4, 0, 0, ...] sequence of labels
            tags: dict["O"] = 4
        Returns:
            list of (chunk_type, chunk_start, chunk_end)
        Example:
            seq = [4, 5, 0, 3]
            tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
            result = [("PER", 0, 2), ("LOC", 3, 4)]
        """
        default = tags["O"]
        idx_to_tag = {idx: tag for tag, idx in tags.items()}
        chunks = []
        chunk_type, chunk_start = None, None
        for i, tok in enumerate(seq):
            # End of a chunk 1
            if tok == default and chunk_type is not None:
                # Add a chunk.
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = None, None
            # End of a chunk + start of a chunk!
            elif tok != default:
                tok_chunk_class, tok_chunk_type = self._get_chunk_type(tok, idx_to_tag)
                if chunk_type is None:
                    chunk_type, chunk_start = tok_chunk_type, i
                elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                    chunk = (chunk_type, chunk_start, i)
                    chunks.append(chunk)
                    chunk_type, chunk_start = tok_chunk_type, i
            else:
                pass
        # end condition
        if chunk_type is not None:
            chunk = (chunk_type, chunk_start, len(seq))
            chunks.append(chunk)
        return chunks
