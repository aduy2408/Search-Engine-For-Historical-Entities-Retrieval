import pandas as pd
import regex as re
from underthesea import word_tokenize
from typing import List, Dict, Set, Literal
import logging
import unicodedata
from unidecode import unidecode
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_to_set_variations(terms: Set[str]) -> Set[str]:
    """Convert a set of underscore-separated terms into all possible variations.
    
    For example: 'thực_hiện' becomes {'thực_hiện', 'thực hiện'}
    
    Args:
        terms: Set of terms, some with underscores
        
    Returns:
        Set of terms with both underscore and space variations
    """
    variations = set()
    for term in terms:
        variations.add(term)  # Add original form
        if '_' in term:
            variations.add(term.replace('_', ' '))  # Add space version
        else:
            variations.add(term.replace(' ', '_'))  # Add underscore version
    return variations

# Cultural terms that should always be preserved
PRESERVE_TERMS_RAW = {
    # Religious and ceremonial places
    'lễ hội', 'di sản', 'văn hóa', 'tín ngưỡng', 'phong tục', 'tập quán',
    'nghi lễ', 'cúng', 'thờ', 'đình', 'chùa', 'miếu', 'đền', 'phủ',
    
    # Cultural heritage terms
    'di tích', 'danh thắng', 'di chỉ', 'cổ vật', 'hiện vật', 'bảo tàng',
    'truyền thống', 'bản sắc', 'dân tộc', 'văn bia', 'bia đá',
    
    # Traditional practices
    'hội làng', 'tế lễ', 'cầu an', 'cầu siêu', 'rước kiệu', 'tục lệ',
    'mai táng', 'thờ cúng', 'tổ tiên', 'thần linh'
}

# Default built-in stop words
DEFAULT_STOP_WORDS_RAW = {
    # Basic stop words
    'và', 'của', 'có', 'được', 'trong', 'các', 'là', 'những', 'cho', 'không',
    'để', 'này', 'khi', 'với', 'về', 'như', 'từ', 'theo', 'tại', 'trên',
    'đã', 'đến', 'sau', 'tới', 'vào', 'rồi', 'thì', 'mà', 'còn', 'nên',
    
    # Cultural heritage specific stop words
    'ngày', 'tháng', 'năm', 'hàng', 'được', 'cùng', 'theo', 'trong', 'ngoài',
    'trước', 'sau', 'đây', 'kia', 'ấy', 'vậy', 'nhất', 'cũng', 'lại', 'mới',
    
    # Temporal markers common in cultural texts
    'xưa', 'nay', 'trước đây', 'hiện nay', 'ngày xưa', 'thời', 'khoảng',
    'triều', 'đời', 'niên', 'kỷ', 'thế kỷ', 'thời kỳ', 'giai đoạn',
    
    # Ceremonial/ritual common words
    'buổi', 'cuộc', 'dịp', 'đợt', 'lần', 'mỗi', 'việc', 'điều', 'cách',
    'hành lễ', 'cử hành', 'tiến hành', 'thực hiện', 'tổ chức', 'diễn ra',
    
    # Historical document markers
    'theo', 'căn cứ', 'dựa vào', 'qua', 'từ đó', 'do đó', 'vì thế',
    'được biết', 'ghi chép', 'tương truyền', 'tục truyền', 'tương tự',
    
    # Measurement units and quantities
    'cái', 'chiếc', 'người', 'con', 'bộ', 'đôi', 'bên', 'phía', 'nhiều',
    'ít', 'vài', 'mấy', 'số', 'khoảng', 'chừng', 'độ', 'phần',
    
    # Location/spatial markers
    'nơi', 'chỗ', 'vùng', 'miền', 'khu vực', 'địa phương', 'vị trí',
    'hướng', 'phía', 'bên', 'trong', 'ngoài', 'trên', 'dưới'
}

# Convert raw sets to include both underscore and space variations
PRESERVE_TERMS = convert_to_set_variations(PRESERVE_TERMS_RAW)
DEFAULT_STOP_WORDS = convert_to_set_variations(DEFAULT_STOP_WORDS_RAW)

def normalize_text_for_comparison(text: str) -> str:
    """Normalize text for comparison by handling spaces and underscores.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text with both space and underscore variations
    """
    # Create variations with both spaces and underscores
    space_version = text.replace('_', ' ')
    underscore_version = text.replace(' ', '_')
    return f"{space_version}|{underscore_version}"

def remove_stop_words(tokens: List[str], stop_words: Set[str]) -> List[str]:
    """Remove stop words while preserving important terms."""
    result = []
    i = 0
    while i < len(tokens):
        # Try to match two-word phrases first
        two_word_match = False
        if i < len(tokens) - 1:
            two_words = f"{tokens[i]} {tokens[i+1]}"
            two_words_normalized = normalize_text_for_comparison(two_words)
            
            # Check if it's a preserve term
            if any(term in two_words_normalized for term in PRESERVE_TERMS):
                result.append(two_words)
                i += 2
                continue
            
            # Check if it's a stop word
            if any(term in two_words_normalized for term in stop_words):
                i += 2
                continue
        
        # If no two-word match, process single word
        if not two_word_match:
            token = tokens[i]
            token_normalized = normalize_text_for_comparison(token)
            
            if (not any(term in token_normalized for term in stop_words) or
                any(term in token_normalized for term in PRESERVE_TERMS) or
                len(token) > 10):  # Preserve long words as they're often meaningful
                result.append(token)
            i += 1
    
    return result

def load_stopwords(mode: Literal['builtin', 'file', 'combined'] = 'builtin', 
                  filepath: str = 'vietnamese-stopwords.txt') -> Set[str]:
    """Load stop words based on specified mode.
    
    Args:
        mode: How to load stop words:
            - 'builtin': Use only built-in stop words
            - 'file': Use only file-based stop words
            - 'combined': Combine both built-in and file-based stop words
        filepath: Path to stop words file
    
    Returns:
        Set of stop words
    """
    file_stop_words = set()
    
    if mode in ['file', 'combined']:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                file_stop_words = {line.strip() for line in f if line.strip()}
            logger.info(f"Loaded {len(file_stop_words)} stop words from file")
        except FileNotFoundError:
            logger.warning(f"Stop words file {filepath} not found.")
            if mode == 'file':
                logger.warning("Falling back to built-in stop words.")
                return DEFAULT_STOP_WORDS
        except Exception as e:
            logger.error(f"Error loading stop words file: {str(e)}")
            if mode == 'file':
                return DEFAULT_STOP_WORDS
    
    if mode == 'builtin':
        logger.info(f"Using {len(DEFAULT_STOP_WORDS)} built-in stop words")
        return DEFAULT_STOP_WORDS
    elif mode == 'file':
        return file_stop_words
    else:  # combined
        combined = DEFAULT_STOP_WORDS.union(file_stop_words)
        logger.info(f"Combined stop words: {len(combined)} total "
                   f"({len(DEFAULT_STOP_WORDS)} built-in + {len(file_stop_words)} from file)")
        return combined

def normalize_unicode(text: str) -> str:
    """Normalize Unicode characters to canonical form."""
    return unicodedata.normalize('NFC', text)

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not isinstance(text, str):
        return ''
    
    # Normalize unicode
    text = normalize_unicode(text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep Vietnamese diacritics
    text = re.sub(r'[^\p{L}\p{N}\s]', ' ', text)
    
    # Remove parentheses and their contents
    text = re.sub(r'\([^)]*\)', '', text)
    
    return text.strip()

def tokenize_and_preserve_terms(text: str) -> List[str]:
    """Tokenize text and preserve important multi-word terms."""
    tokens = word_tokenize(text)
    final_tokens = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1:
            bigram = tokens[i] + '_' + tokens[i+1]
            if bigram in PRESERVE_TERMS:
                final_tokens.append(bigram)
                i += 2
                continue
        final_tokens.append(tokens[i])
        i += 1
    return final_tokens

def preprocess_text(text: str, stop_words: Set[str]) -> Dict[str, str]:
    """Process a single text entry.
    
    Args:
        text: Input text to process
        stop_words: Set of stop words to use
    
    Returns:
        Dictionary containing processed versions of the text
    """
    cleaned_text = clean_text(text)
    tokens = tokenize_and_preserve_terms(cleaned_text)
    tokens = remove_stop_words(tokens, stop_words)
    
    processed_text = ' '.join(tokens)
    return {
        'text_with_diacritics': processed_text,
        'text_without_diacritics': unidecode(processed_text),
        'token_count': len(tokens)
    }

def process_dataframe(df: pd.DataFrame, text_columns: List[str], 
                     mode: Literal['builtin', 'file', 'combined'] = 'builtin') -> pd.DataFrame:
    """Process multiple text columns in a dataframe.
    
    Args:
        df: Input dataframe
        text_columns: List of column names containing text to process
        mode: Which stop words to use ('builtin', 'file', or 'combined')
    
    Returns:
        Processed dataframe with new columns
    """
    stop_words = load_stopwords(mode)
    processed_df = df.copy()
    
    for column in text_columns:
        if column not in df.columns:
            logger.warning(f"Column {column} not found in dataframe")
            continue
            
        logger.info(f"Processing column: {column}")
        
        # Process each text entry
        processed_texts = [preprocess_text(text, stop_words) for text in df[column]]
        
        # Add new columns for processed text
        processed_df[f"{column}_processed"] = [text['text_with_diacritics'] for text in processed_texts]
        processed_df[f"{column}_normalized"] = [text['text_without_diacritics'] for text in processed_texts]
        processed_df[f"{column}_token_count"] = [text['token_count'] for text in processed_texts]
    
    return processed_df

if __name__ == "__main__":
    try:
        # Load the sample CSV file
        df = pd.read_csv('sample_csv.csv')
        logger.info(f"Loaded CSV file with {len(df)} rows")
        
        # Process with different stop words options
        text_columns = ['title', 'content']
        
        # Using built-in stop words
        logger.info("Processing with built-in stop words...")
        processed_df_builtin = process_dataframe(df, text_columns, mode='builtin')
        processed_df_builtin.to_csv('processed_vietnamese_texts_builtin.csv', index=False)
        
        # Using file-based stop words
        logger.info("Processing with file-based stop words...")
        processed_df_file = process_dataframe(df, text_columns, mode='file')
        processed_df_file.to_csv('processed_vietnamese_texts_file.csv', index=False)
        
        # Using combined stop words
        logger.info("Processing with combined stop words...")
        processed_df_combined = process_dataframe(df, text_columns, mode='combined')
        processed_df_combined.to_csv('processed_vietnamese_texts_combined.csv', index=False)
        
        # Log statistics about stop words
        builtin_words = load_stopwords('builtin')
        file_words = load_stopwords('file')
        combined_words = load_stopwords('combined')
        
        logger.info(f"Built-in stop words count: {len(builtin_words)}")
        logger.info(f"File-based stop words count: {len(file_words)}")
        logger.info(f"Combined stop words count: {len(combined_words)}")
        logger.info(f"Unique words added from file: {len(combined_words - builtin_words)}")
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}") 