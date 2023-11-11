# This file is a module for the actual AI. It is meant to preprocess the data

_prep_tokens = [',', '.', '"', "'", '<br>', "<i>", "</i>", "(", ")", "[", "]", "!", "?", ":", ";", " ", "\n", "\t", "<br/>"]

class Preprocessor:
    
    def tokenize(data):
        _t_data = data.split()
        return _t_data
    
    class preprocess:
        
        def words(data):
            for token in _prep_tokens:
                data = data.replace(token, " ")
            data = data.lower()
            data = Preprocessor.tokenize(data)
            return data
        
        def characters(data):
            _data = []
            for token in _prep_tokens:
                data = data.replace(token, " ")
            for letter in data:
                if letter.isdigit():
                    data = data.replace(letter, " ")
                else:
                    _data.append(letter)
            
            return _data
    
    class Vocabulary:
        
        class create_vocab:
            def sequences(preprocessed_data):
                _current_seq = 1
                _vocab = {}
                for token in preprocessed_data:
                    if token not in _vocab:
                        _vocab[token] = _current_seq
                        _current_seq += 1
                return _vocab
        
if __name__ == "__main__":
    tokenized_data = Preprocessor.preprocess.words("The quick brown fox jumps over the lazy dog, but not without first stopping to admire the vibrant colors of the blooming flowers, feeling the warmth of the sun on its fur, and listening to the melodic chirping of the birds in the distance.")
    print("Tokenized data: ", tokenized_data)
    vocab = Preprocessor.Vocabulary.create_vocab.sequences(tokenized_data)
    print("Vocabulary: ", vocab)
    print("Vocabulary length: ", len(vocab))