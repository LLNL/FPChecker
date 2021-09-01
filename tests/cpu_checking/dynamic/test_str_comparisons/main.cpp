
int main() {
  int case_1 = _FPC_STRING_ENDS_WITH("hello you", "you");
  int case_2 = _FPC_STRING_ENDS_WITH("file.cpp", "file.cpp");
  int case_3 = _FPC_STRING_ENDS_WITH("/path/to/file.cpp", "file.cpp");
  int case_4 = _FPC_STRING_ENDS_WITH("file.cpp", "");
  int case_5 = _FPC_STRING_ENDS_WITH("", "");
  
  int case_6 = _FPC_STRING_ENDS_WITH("file.cpp", "/file.cpp");
  int case_7 = _FPC_STRING_ENDS_WITH("b/file.cpp", "a/file.cpp");
  int case_8 = _FPC_STRING_ENDS_WITH("/path/file.cpp", "path/file.c");


  printf("case_1 %d\n", case_1);
  printf("case_2 %d\n", case_2);
  printf("case_3 %d\n", case_3);
  printf("case_4 %d\n", case_4);
  printf("case_5 %d\n", case_5);
  printf("case_6 %d\n", case_6);
  printf("case_7 %d\n", case_7);
  printf("case_8 %d\n", case_8);
}
