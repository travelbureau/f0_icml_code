#ifndef __ZHELPERS_HPP_INCLUDED__
#define __ZHELPERS_HPP_INCLUDED__ 
#include <zmq.hpp>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <time.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <signal.h>
#if (!defined(WIN32))
# include <sys/time.h>
# include <unistd.h>
#endif
#if (defined (WIN32))
    typedef unsigned long ulong;
    typedef unsigned int uint;
    typedef __int64 int64_t;
#endif
#if (defined (WIN32))
#define srandom srand
#define random rand
#endif
#if defined(_MSC_VER) && _MSC_VER < 1900
#define snprintf c99_snprintf
#define vsnprintf c99_vsnprintf
 inline int c99_vsnprintf(char *outBuf, size_t size, const char *format, va_list ap)
 {
  int count = -1;
  if (size != 0)
   count = _vsnprintf_s(outBuf, size, _TRUNCATE, format, ap);
  if (count == -1)
   count = _vscprintf(format, ap);
  return count;
 }
 inline int c99_snprintf(char *outBuf, size_t size, const char *format, ...)
 {
  int count;
  va_list ap;
  va_start(ap, format);
  count = c99_vsnprintf(outBuf, size, format, ap);
  va_end(ap);
  return count;
 }
#endif
#define within(num) (int) ((float) (num) * random () / (RAND_MAX + 1.0))
static std::string
s_recv (zmq::socket_t & socket) {
    zmq::message_t message;
    socket.recv(&message);
    return std::string(static_cast<char*>(message.data()), message.size());
}
static bool
s_send (zmq::socket_t & socket, const std::string & string) {
    zmq::message_t message(string.size());
    memcpy (message.data(), string.data(), string.size());
    bool rc = socket.send (message);
    return (rc);
}
static bool
s_sendmore (zmq::socket_t & socket, const std::string & string) {
    zmq::message_t message(string.size());
    memcpy (message.data(), string.data(), string.size());
    bool rc = socket.send (message, ZMQ_SNDMORE);
    return (rc);
}
static void
s_dump (zmq::socket_t & socket)
{
    std::cout << "----------------------------------------" << std::endl;
    while (1) {
        zmq::message_t message;
        socket.recv(&message);
        int size = message.size();
        std::string data(static_cast<char*>(message.data()), size);
        bool is_text = true;
        int char_nbr;
        unsigned char byte;
        for (char_nbr = 0; char_nbr < size; char_nbr++) {
            byte = data [char_nbr];
            if (byte < 32 || byte > 127)
                is_text = false;
        }
        std::cout << "[" << std::setfill('0') << std::setw(3) << size << "]";
        for (char_nbr = 0; char_nbr < size; char_nbr++) {
            if (is_text)
                std::cout << (char)data [char_nbr];
            else
                std::cout << std::setfill('0') << std::setw(2)
                   << std::hex << (unsigned int) data [char_nbr];
        }
        std::cout << std::endl;
        int more = 0;
        size_t more_size = sizeof (more);
        socket.getsockopt (ZMQ_RCVMORE, &more, &more_size);
        if (!more)
            break;
    }
}
#if (!defined (WIN32))
inline std::string
s_set_id (zmq::socket_t & socket)
{
    std::stringstream ss;
    ss << std::hex << std::uppercase
       << std::setw(4) << std::setfill('0') << within (0x10000) << "-"
       << std::setw(4) << std::setfill('0') << within (0x10000);
    socket.setsockopt(ZMQ_IDENTITY, ss.str().c_str(), ss.str().length());
    return ss.str();
}
#else
inline std::string
s_set_id(zmq::socket_t & socket, intptr_t id)
{
    std::stringstream ss;
    ss << std::hex << std::uppercase
        << std::setw(4) << std::setfill('0') << id;
    socket.setsockopt(ZMQ_IDENTITY, ss.str().c_str(), ss.str().length());
    return ss.str();
}
#endif
static void
s_version (void)
{
    int major, minor, patch;
    zmq_version (&major, &minor, &patch);
    std::cout << "Current 0MQ version is " << major << "." << minor << "." << patch << std::endl;
}
static void
s_version_assert (int want_major, int want_minor)
{
    int major, minor, patch;
    zmq_version (&major, &minor, &patch);
    if (major < want_major
    || (major == want_major && minor < want_minor)) {
        std::cout << "Current 0MQ version is " << major << "." << minor << std::endl;
        std::cout << "Application needs at least " << want_major << "." << want_minor
              << " - cannot continue" << std::endl;
        exit (EXIT_FAILURE);
    }
}
static int64_t
s_clock (void)
{
#if (defined (WIN32))
 FILETIME fileTime;
 GetSystemTimeAsFileTime(&fileTime);
 unsigned __int64 largeInt = fileTime.dwHighDateTime;
 largeInt <<= 32;
 largeInt |= fileTime.dwLowDateTime;
 largeInt /= 10000;
 return (int64_t)largeInt;
#else
    struct timeval tv;
    gettimeofday (&tv, NULL);
    return (int64_t) (tv.tv_sec * 1000 + tv.tv_usec / 1000);
#endif
}
static void
s_sleep (int msecs)
{
#if (defined (WIN32))
    Sleep (msecs);
#else
    struct timespec t;
    t.tv_sec = msecs / 1000;
    t.tv_nsec = (msecs % 1000) * 1000000;
    nanosleep (&t, NULL);
#endif
}
static void
s_console (const char *format, ...)
{
    time_t curtime = time (NULL);
    struct tm *loctime = localtime (&curtime);
    char *formatted = new char[20];
    strftime (formatted, 20, "%y-%m-%d %H:%M:%S ", loctime);
    printf ("%s", formatted);
    delete[] formatted;
    va_list argptr;
    va_start (argptr, format);
    vprintf (format, argptr);
    va_end (argptr);
    printf ("\n");
}
static int s_interrupted = 0;
static void s_signal_handler (int signal_value)
{
    s_interrupted = 1;
}
static void s_catch_signals ()
{
#if (!defined(WIN32))
    struct sigaction action;
    action.sa_handler = s_signal_handler;
    action.sa_flags = 0;
    sigemptyset (&action.sa_mask);
    sigaction (SIGINT, &action, NULL);
    sigaction (SIGTERM, &action, NULL);
#endif
}
#endif
