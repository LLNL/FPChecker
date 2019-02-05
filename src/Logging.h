/*
 * Logging.h
 *
 *  Created on: Feb 4, 2019
 *      Author: Ignacio Laguna, ilaguna@llnl.gov
 */

#ifndef SRC_LOGGING_H_
#define SRC_LOGGING_H_

namespace CUDAAnalysis {

#define FPC_PREFIX "#FPCHECKER: "

class Logging
{

public:
	static void info(const char *msg);
	static void error(const char *msg);
};

}




#endif /* SRC_LOGGING_H_ */
