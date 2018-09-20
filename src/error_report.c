/*
 * error_report.c
 *
 *  Created on: Sep 18, 2018
 *      Author: Ignacio Laguna, ilaguna@llnl.gov
 */

#include <stdio.h>
#include <string.h>

#define REPORT_LINE_SIZE 80
#define REPORT_COL1_SIZE 15
#define REPORT_COL2_SIZE REPORT_LINE_SIZE-REPORT_COL1_SIZE-1

///
/// .------------------------------------------------.
/// | FPChecker Error Report                         |
/// |------------------------------------------------|
/// | Error     | NaN                                |
/// | Operation | DIV                                |
/// | File      | ForceCalculation.cc                |
/// | Line      | 23                                 |
/// | Thread ID | 1024                               |
/// '------------------------------------------------'
///

/// .----------------------------------------------------------.
/// |---------------- FPChecker Error Report ------------------|
///  Error     | NaN
///  Operation | DIV
///  File      | /usr/local/bin/long/path/to/file/file/pathForceCalculation.cc
///  Line      | 23
///  Thread ID | 1024
/// '----------------------------------------------------------'

/// .----------------------------------------------------------.
/// |---------------- FPChecker Error Report ------------------|
///  Error     | NaN
///  Operation | DIV
///  File      | /usr/file/pathForceCalculation.cc
///  Line      | 23
///  Thread ID | 1024
/// '----------------------------------------------------------'

void _FPC_PRINT_REPORT_LINE_(const char border)
{
	printf("%c",border);
	for (int i=0; i < REPORT_LINE_SIZE-2; ++i)
		printf("-");
	printf("%c\n",border);
}

void _FPC_PRINT_REPORT_HEADER_(int type)
{
	_FPC_PRINT_REPORT_LINE_('.');

	char msg[255];
	msg[0] = '\0';
	if (type == 0)
		strcpy(msg,"| FPChecker Error Report");
	else
		strcpy(msg,"| FPChecker Warning Report");
	printf("%s",msg);

	int rem = strlen(msg);
	for (int i=0; i < REPORT_LINE_SIZE-rem-1; ++i)
		printf(" ");
	printf("|\n");

	_FPC_PRINT_REPORT_LINE_('|');
}

void _FPC_PRINT_REPORT_ROW_(const char *val, int space, int last)
{
	char msg[255];
	msg[0] = '\0';
	strcpy(msg,"| ");
	strcat(msg, val);
	int rem = strlen(msg);
	for (int i=0; i < space-rem; ++i)
		strcat(msg," ");
	printf("%s",msg);

	if (last==1)
		printf("|\n");
}

int main()
{

	_FPC_PRINT_REPORT_HEADER_(1);
	_FPC_PRINT_REPORT_ROW_("Error", REPORT_COL1_SIZE, 0);
	_FPC_PRINT_REPORT_ROW_("NaN", REPORT_COL2_SIZE, 1);
	_FPC_PRINT_REPORT_ROW_("Operation", REPORT_COL1_SIZE, 0);
	_FPC_PRINT_REPORT_ROW_("DIV", REPORT_COL2_SIZE, 1);
	_FPC_PRINT_REPORT_ROW_("File", REPORT_COL1_SIZE, 0);
	_FPC_PRINT_REPORT_ROW_("This Is a long file", REPORT_COL2_SIZE, 1);
	_FPC_PRINT_REPORT_LINE_('\'');

	return 0;
}
