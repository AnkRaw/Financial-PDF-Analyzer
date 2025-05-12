from typing import List, Any, Tuple, Dict
from pydantic import BaseModel
from unstructured.documents.elements import (
    Title, Header, NarrativeText, Text, ListItem, Table, Image, FigureCaption, Formula
)

class Element(BaseModel):
    type: str
    text: str
    metadata: Dict[str, Any] = {}

class DocumentPreprocessor:
    def __init__(self, raw_elements: List[Any]):
        self.raw_elements = raw_elements
        self.chunks: List[Element] = []
        self.current_chunk = ""
        self.current_metadata: List[Dict[str, Any]] = []

    def _flatten_metadata(self, metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        def safe_get(d, path, default=None):
            for p in path:
                d = d.get(p, {}) if isinstance(d, dict) else {}
            return d or default
        
        types = []
        element_ids = []
        page_numbers = []
        languages = set()
        parent_ids = []
        source_indexes = []

        for m in metadata_list:
            # types.append(m.get("type", ""))
            # element_ids.append(m.get("element_id", ""))
            # page_numbers.append(str(safe_get(m, ["metadata", "page_number"], "")))
            # parent_ids.append(str(safe_get(m, ["metadata", "parent_id"], "")))
            # source_indexes.append(str(m.get("source_index", "")))
            langs = safe_get(m, ["metadata", "languages"], "")
            if isinstance(langs, list):
                languages.add(",".join(map(str, langs)))

        return {
            # "types": ",".join(filter(None, types)),
            # "element_ids": ",".join(filter(None, set(element_ids))),
            # "page_numbers": ",".join(filter(None, set(page_numbers))),
            "languages": ",".join(sorted(languages)),
            # "parent_ids": ",".join(filter(None, set(parent_ids))),
            # "source_indexs": ",".join(filter(None, set(source_indexes))),
        }

    def _flush_chunk(self, index: int = -1):
        if self.current_chunk.strip():
            metadata = self._flatten_metadata(self.current_metadata)
            # print(metadata)
            self.chunks.append(Element(type="text", text=self.current_chunk.strip(), metadata=metadata))
            self.current_chunk = ""
            self.current_metadata = []
    
    def extract_element_metadata(self, element):
        element_dict = element.to_dict()
        
        # Extract each key safely and skip missing values
        metadata = {}
        # keys_to_extract = ["type", "element_id", "page_number", "languages", "parent_id"]
        keys_to_extract = ["type", "languages"]
        
        for key in keys_to_extract:
            try:
                value = None
                if key in ["type", "element_id"]:
                    value = element_dict.get(key, None)
                else:
                    value = element_dict.get("metadata", {}).get(key, None)
                
                # Convert 'languages' list to comma-separated string
                if key == "languages" and isinstance(value, list):
                    value = ", ".join(value)
                elif value is None:
                    value = ""
                
                metadata[key] = value
            except (KeyError, TypeError):
                metadata[key] = ""
                continue
        
        return metadata
    
    def preprocess(self) -> List[Element]:
        i = 0
        while i < len(self.raw_elements):
            element = self.raw_elements[i]

            # Group Headers and Titles
            if isinstance(element, (Title, Header)):
                self._flush_chunk(index=i)
                self.current_chunk += f"\n\n{element.text.strip()}\n"
                e_metadata = element.to_dict()
                e_metadata["source_index"] = i
                self.current_metadata.append(e_metadata)
                # print("Title: ", self.current_metadata)
                i += 1
                while i < len(self.raw_elements) and isinstance(self.raw_elements[i], (Title, Header)):
                    self.current_chunk += f"{self.raw_elements[i].text.strip()}\n"
                    e_metadata = self.raw_elements[i].to_dict()
                    e_metadata["source_index"] = i
                    self.current_metadata.append(e_metadata)
                    # print("Title while: ", self.current_metadata)
                    i += 1
                continue

            # Add textual elements  
            elif isinstance(element, (NarrativeText, Text, ListItem, Image, FigureCaption, Formula)):
                self.current_chunk += f"{element.text.strip()}\n"
                e_metadata = element.to_dict()
                e_metadata["source_index"] = i
                self.current_metadata.append(e_metadata)
                # print("other: ", self.current_metadata)

            # Handle tables
            if isinstance(element, Table):
                self._flush_chunk(index=i)
                table_html = element.to_dict().get("metadata", {}).get("text_as_html", "")
                metadata = self.extract_element_metadata(element)
                metadata['source_index'] = i
                self.chunks.append(Element(type="table", text=table_html, metadata=metadata))

            i += 1

        self._flush_chunk(index=i)
        return self.chunks

    def split_by_type(self) -> Tuple[List[Element], List[Element]]:
        table_elements = [e for e in self.chunks if e.type == "table"]
        text_elements = [e for e in self.chunks if e.type == "text"]
        return text_elements, table_elements
    
    def preprocess_as_html(self) -> List[Element]:
        self.chunks = []
        self.current_chunk = ""
        self.current_metadata = []
        keys_to_keep = ["type", "source_index", "languages", "page_number"]
        
        tag_map = {
            "Header": "H",
            "Title": "T",
            "NarrativeText": "NT",
            "Image": "IM",
            "Table": "TB",
            "Text": "TX",
            "ListItem": "LI",
            "FigureCaption": "FC",
            "Formula": "F"
        }

        def get_tag(class_name: str) -> str:
            return tag_map.get(class_name, class_name)

        def format_element(element, index: int) -> str:
            class_name = element.__class__.__name__
            tag = get_tag(class_name)
            element_dict = element.to_dict()
            text = element.text.strip() if hasattr(element, 'text') else ""

            # Use table HTML if applicable
            if class_name == "Table":
                text = element_dict.get("metadata", {}).get("text_as_html", "").strip()

            page_number = element_dict.get("metadata", {}).get("page_number", "")
            page_suffix = f"[P{page_number}]" if page_number else ""

            return f"<{tag}>{text}</{tag}>{page_suffix}"

        i = 0
        while i < len(self.raw_elements):
            element = self.raw_elements[i]
            class_name = element.__class__.__name__
            # print(class_name)
            
            # Handle tables as separate Element
            if isinstance(element, Table):
                self._flush_chunk(index=i)
                table_html = element.to_dict().get("metadata", {}).get("text_as_html", "")
                metadata = element.metadata.to_dict()
                metadata = {k: metadata[k] for k in keys_to_keep if k in metadata.keys()}
                metadata['languages'] = ",".join(metadata['languages'] )
                # metadata['source_index'] = i
                self.chunks.append(Element(type="table", text=table_html, metadata=metadata))
            
            # Start a new chunk for Title/Header groups
            if isinstance(element, (Title, Header)):
                self._flush_chunk()
                while i < len(self.raw_elements) and isinstance(self.raw_elements[i], (Title, Header)):
                    html_str = format_element(self.raw_elements[i], i)
                    self.current_chunk += html_str + " "
                    e_metadata = self.raw_elements[i].to_dict()
                    # e_metadata["source_index"] = i
                    self.current_metadata.append(e_metadata)
                    i += 1
                continue

            # Normal text-like elements
            elif isinstance(element, (NarrativeText, Text, ListItem, Image, FigureCaption, Formula)):
                html_str = format_element(element, i)
                self.current_chunk += html_str + " "
                e_metadata = element.to_dict()
                # e_metadata["source_index"] = i
                self.current_metadata.append(e_metadata)

            i += 1

        self._flush_chunk()
        return self.chunks


class Chunker:
    def __init__(self, chunk_size, overlap):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _count_tokens(self, text: str) -> int:
        return len(text.split())  
    
    def _flatten_metadata(self, metadata_list: List[dict]) -> dict:
        result = {}

        for m in metadata_list:
            for key, value in m.items():
                if value is None:
                    continue

                if key not in result:
                    if isinstance(value, list):
                        result[key] = list(value)
                    elif isinstance(value, str):
                        result[key] = [value]
                    else:
                        result[key] = [str(value)]
                else:
                    if isinstance(value, list):
                        result[key].extend(value)
                    elif isinstance(value, str):
                        result[key].append(value)
                    else:
                        result[key].append(str(value))

        # Deduplicate and join values as appropriate
        for key in result:
            result[key] = ",".join(sorted(set(result[key])))

        return result


    def chunk_elements(self, elements: List[Element]) -> List[Element]:
        chunks: List[Element] = []
        current_tokens = []
        current_length = 0
        current_metadata = []
        i = 0

        while i < len(elements):
            el = elements[i]
            tokens = el.text.split()
            token_len = len(tokens)

            if current_length + token_len > self.chunk_size:
                chunk_text = " ".join(current_tokens)
                chunk_metadata = {"source_type": "text", **self._flatten_metadata(current_metadata)}
                chunks.append(Element(type="text", text=chunk_text, metadata=chunk_metadata))


                if self.overlap > 0:
                    overlap_tokens = current_tokens[-self.overlap:] if len(current_tokens) >= self.overlap else current_tokens
                    current_tokens = overlap_tokens[:]
                    current_length = len(current_tokens)
                    current_metadata = current_metadata[-self.overlap:]
                else:
                    current_tokens = []
                    current_length = 0
                    current_metadata = []

            current_tokens.extend(tokens)
            current_metadata.append(el.metadata)
            current_length += token_len
            i += 1

        if current_tokens:
            chunk_text = " ".join(current_tokens)
            chunk_metadata = {"source_type": "text", **self._flatten_metadata(current_metadata)}
            chunks.append(Element(type="text", text=chunk_text, metadata=chunk_metadata))


        return chunks
