#ifndef WRAPPER_BOOST_ARGPARSER
#define WRAPPER_BOOST_ARGPARSER

#include <iostream>
#include <assert.h>
#include <boost/program_options.hpp>

// How to compile?  g++ -o $TARGET $SRC -lboost_program_options 
// Reference : http://www.boost.org/doc/libs/1_63_0/doc/html/program_options.html

// ex) ./$TARGET --option1=value1 --option2=value2 ....

class boost_argparser
{
public:
	boost_argparser(int Argc, char** Argv)
	: _Argc(Argc), _Argv(Argv), _desc("Allowed options"), isDataStored(false) {}

	void add_option(const char optionName[], const char Description[] = "None")
	{
		_desc.add_options()
			(optionName, boost::program_options::value<std::string>(), Description);
	}

	void store()
	{
		boost::program_options::store(boost::program_options::parse_command_line(_Argc, _Argv, _desc), _vm);
		isDataStored = true;
	}

	bool isOptionExist(const char optionName[]) const
	{
		assert(isDataStored);
		return _vm.count(optionName);
	}

	std::string operator[](const char optionName[]) const
	{
		std::string value("THEREISNOMACHEDVALUE!");
		assert(isDataStored);
		assert(isOptionExist(optionName));
		if(isOptionExist(optionName))
			value = (_vm[optionName].as<std::string>());
		return value;
	}

	void load_help_option() const
	{
		std::cout<<_desc<<std::endl;
	}

private:
	int _Argc;
	char** _Argv;
	boost::program_options::options_description _desc;
	bool isDataStored;
	boost::program_options::variables_map _vm;
};

#endif
