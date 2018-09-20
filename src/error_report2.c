/*
 * error_report.c
 *
 *  Created on: Sep 18, 2018
 *      Author: Ignacio Laguna, ilaguna@llnl.gov
 */

#include <stdio.h>
//#include <string.h>

#define REPORT_LINE_SIZE 80
#define REPORT_COL1_SIZE 15
#define REPORT_COL2_SIZE REPORT_LINE_SIZE-REPORT_COL1_SIZE-1

/// +---------------- FPChecker Error Report ------------------+
///  Error     : NaN
///  Operation : DIV
///  File      : /usr/file/pathForceCalculation.cc
///  Line      : 23
///  Thread ID : 1024
/// +----------------------------------------------------------+

int _FPC_LEN_(const char *s)
{
	int maxLen = 1024; // to check correctness and avid infinite loop
	int i = 0;
	while(s[i] != '\0' && i < maxLen)
		i++;
	return i;
}

void _FPC_CPY_(char *d, const char *s)
{
	int len = _FPC_LEN_(s);
	int i=0;
	for (i=0; i < len; ++i)
		d[i] = s[i];
	d[i] = '\0';
}

void _FPC_CAT_(char *d, const char *s)
{
	int lenS = _FPC_LEN_(s);
	int lenD = _FPC_LEN_(d);
	int i=0;
	for (i=0; i < lenS; ++i)
		d[i+lenD] = s[i];
	d[i+lenD] = '\0';
}

void _FPC_PRINT_REPORT_LINE_(const char border)
{
	printf("%c",border);
	for (int i=0; i < REPORT_LINE_SIZE-2; ++i)
		printf("-");
	printf("%c\n",border);
}

void _FPC_PRINT_REPORT_HEADER_(int type)
{
	//_FPC_PRINT_REPORT_LINE_('.');

	char msg[255];
	msg[0] = '\0';
	if (type == 0)
		_FPC_CPY_(msg," FPChecker Error Report ");
	else
		_FPC_CPY_(msg," FPChecker Warning Report ");

	int l = _FPC_LEN_(msg);
	l = REPORT_LINE_SIZE-l-2;
	char line[255];
	line[0] = '\0';
	_FPC_CAT_(line,"+");
	for (int i=0; i < l/2; ++i)
		_FPC_CAT_(line,"-");
	if (l%2)
		_FPC_CAT_(line,"-");
	_FPC_CAT_(line,msg);
	for (int i=0; i < l/2; ++i)
		_FPC_CAT_(line,"-");
	_FPC_CAT_(line,"+");
	printf("%s\n",line);
}

void _FPC_PRINT_REPORT_ROW_(const char *val, int space, int last)
{
	char msg[255];
	msg[0] = '\0';
	_FPC_CPY_(msg," ");
	_FPC_CAT_(msg, val);
	int rem = _FPC_LEN_(msg);
	for (int i=0; i < space-rem; ++i)
		_FPC_CAT_(msg," ");
	printf("%s",msg);

	if (last==0)
		printf(":");
	else
		printf("\n");
}

int main()
{

	_FPC_PRINT_REPORT_HEADER_(0);
	_FPC_PRINT_REPORT_ROW_("Error", REPORT_COL1_SIZE, 0);
	_FPC_PRINT_REPORT_ROW_("NaN", REPORT_COL2_SIZE, 1);
	_FPC_PRINT_REPORT_ROW_("Operation", REPORT_COL1_SIZE, 0);
	_FPC_PRINT_REPORT_ROW_("DIV", REPORT_COL2_SIZE, 1);
	_FPC_PRINT_REPORT_ROW_("File", REPORT_COL1_SIZE, 0);
	_FPC_PRINT_REPORT_ROW_("/usr/local/bin/long/path/to/file/file/pathForceCalculation.cc", REPORT_COL2_SIZE, 1);
	_FPC_PRINT_REPORT_LINE_('+');

	return 0;
}
