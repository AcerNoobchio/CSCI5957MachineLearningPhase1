class StringUtil:
    """Contains string methods"""
    #Extracts a substring to the right of a given delimiter - finds the delimiter looking backwards
    @staticmethod
    def rExtractSubstring(stringIn, Delimiter):
        lastIndex = stringIn.rfind(Delimiter)
        lastIndex += len(Delimiter)
        newString = stringIn[lastIndex:]
        return newString
    #Extracts a substring to the left of a given delimiter - finds the delimiter by searching from the beginning of the string
    @staticmethod
    def extractSubstring(stringIn, Delimiter):
        lastIndex = stringIn.rfind(Delimiter)
        newString = stringIn[:lastIndex]
        return newString
    #Extracts a filename from a passed file path
    @staticmethod
    def extractFileName(filepathIn):
        return StringUtil.extractSubstring(StringUtil.rExtractSubstring(filepathIn, "\\"), ".")

