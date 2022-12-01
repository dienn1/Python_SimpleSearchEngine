from urllib.parse import urldefrag
import zlib


class FilterDuplicate:
    def __init__(self, checksum=None, urls=None):
        self.content_checksum = set() if checksum is None else checksum
        self.unique_urls = set() if urls is None else urls

    # Return True if no duplicate
    def add_tokens(self, tokens_list):
        crc_checksum = zlib.crc32(bytes(" ".join(tokens_list), "utf-8"))
        if crc_checksum in self.content_checksum:
            return False
        self.content_checksum.add(crc_checksum)
        return True

    # Return defrag url if no duplicate, None otherwise
    def add_url(self, url):
        url = urldefrag(url).url
        if url in self.unique_urls:
            return None
        self.unique_urls.add(url)
        return url
