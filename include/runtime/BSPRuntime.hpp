/*
 * BSPRuntime.hpp
 *
 *  Created on: 2014-7-9
 *      Author: junfeng
 */

#ifndef BSPRUNTIME_HPP_
#define BSPRUNTIME_HPP_

#include <stdint.h>
#include <string>
#include <mpi.h>
#include <map>
#include <vector>
#include <queue>
#include <sstream>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>
#include "BSPNet.hpp"
#include "BSPGrid.hpp"
#include "BSPException.hpp"
#include "BSPNameSpace.hpp"
#include "BSPIndexSet.hpp"
namespace BSP {
    class ArrayRegistration;
    class GlobalArray;
    class LocalArray;
    class GlobalRequest;
    class LocalRequest;
    class Message;
    class Program;
    enum Type;

    class ASyncScheduler {
        private:
            uint64_t _iter, _nPassengers, _minPlanSize, _maxPlanSize, _nPlans;
            uint64_t *_deadline;
            uint64_t *_ticket;
            uint64_t *_buffer;
            uint64_t **_plan;
            bool *_arrived;
        protected:
            void finalize();
            bool reschedule(uint64_t i);
        public:
            ASyncScheduler(uint64_t nPassengers);
            ASyncScheduler(uint64_t nPassengers, uint64_t nPlans);
            ASyncScheduler(uint64_t nPassengers, uint64_t nPlans, uint64_t minPlanSize, uint64_t maxPlanSize);
            ~ASyncScheduler();
            void init(uint64_t nPassengers, uint64_t nPlans);
            void init(uint64_t nPassengers, uint64_t nPlans, uint64_t minPlanSize, uint64_t maxPlanSize);
            void receive(uint64_t i);
            bool ready();
            bool complete();
            void iterate();
            inline uint64_t sizeOfManifest() { return _plan[0][_maxPlanSize]; }
            inline uint64_t itemOfManifest(uint64_t j) { return _plan[0][j]; }
            inline uint64_t ticketOf(uint64_t j) { return _ticket[j]; }
    };

    /// @brief the main class of C$ runtime library
    class Runtime {
        friend class ArrayRegistration;
    private:
        Net _nal;
        Grid _grid;
        uint64_t _nProcs;
        uint64_t _myProcID;
        uint64_t _indexOfGridDim[7];
        std::string _locationString;
	int _currentSubRank;
        ArrayRegistration *_registration;
        NameSpace _this;
        uint64_t *_nOutgoingRequestsAndUpdates;
        uint64_t *_nIncomingReplies;
        std::vector<Message *> *_incomingUserDefinedMessages; // reserved for later use
        std::vector<Message *> *_outgoingUserDefinedMessages;
        std::queue<std::string> _eventQueue;
        std::vector<std::string> _syncList;
        std::map<GlobalRequest *, LocalArray *> _replyReceivers;
        std::stringstream *_incomingUserArrayNames; // reserved for later use
        std::stringstream *_outgoingUserArrayNames;
        std::map<uint64_t, uint64_t> _workerID;
        std::vector<uint64_t> _workerProc;
        std::vector<uint64_t> _manifest;
        ASyncScheduler *_scheduler;
        bool _scheduling;
        static Runtime *_activeRuntimeObject;
        bool _verbose;
        bool _finalizing;


        void fillMyMessageHeader(uint64_t& partnerID, uint64_t*& myMessageHeader);
        void exchangeMessageHeaders(uint64_t& myReqUpdCount,
                uint64_t*& myMessageHeader, uint64_t& partnerID,
                uint64_t& partnerReqUpdCount, uint64_t*& partnerMessageHeader);
        std::vector<Message*> exchangeRequestsAndUpdates(uint64_t myReqUpdCount,
                uint64_t& partnerID, uint64_t& partnerReqUpdCount,
                uint64_t* partnerMessageHeader);
        uint64_t receiveMyReplies(uint64_t& partnerID);
        void sendRepliesToPartner(uint64_t& partnerReqUpdCount,
                std::vector<Message*>& partnerMessages, uint64_t& partnerID,
                std::vector<Message*>& repliesToPartner);
        void copyUserDefinedArrays(uint64_t& partnerReqUpdCount,
                std::vector<Message*>& partnerMessages, uint64_t partnerID);
        void applyUpdates(uint64_t& partnerReqUpdCount,
                std::vector<Message*>& partnerMessages);
        std::string getLastTokenOfPath(std::string path);
        void checkNameList(std::vector<std::string> nameList, bool checkShared);

    public:

        /// @brief constructor
        /// @param pArgc the pointer to argc of main()
        /// @param pArgv the pointer to argv of main()
        Runtime(int *pArgc, char ***pArgv);

        /// @brief destructor
        ~Runtime();

        inline bool inGrid() {
            return _grid.containsProc(_myProcID);
        }

        Grid &getGrid() {
            return _grid;
        }

        /// @brief get number of processes
        /// @return the number of processes

        uint64_t getNumberOfProcesses() {
            return _nProcs;
        }

        /// @brief get my process ID
        /// @return the ID of my process

        uint64_t getMyProcessID() {
            return _myProcID;
        }

        std::string getLocation() {
            return _locationString;
        }

        /// @brief get the active runtime object

        static Runtime *getActiveRuntime() {
            return _activeRuntimeObject;
        }

        /// @brief get "this" namespace

        NameSpace &getThis() {
            return _this;
        }

        /// @brief get nal

        Net *getNAL() {
            return &_nal;
        }


        /// @brief get the size of the manifest
        inline uint64_t sizeOfManifest() { return _manifest.size(); }

        /// @brief get one item of the manifest
        inline uint64_t itemOfManifest(uint64_t j) { return _manifest[j]; }

        /// @brief abort
        void abort();

        /// @brief resume a program
        ///void resumeProgram(std::string programID);

        /// @brief request data from/to global array
        void requestFrom(GlobalArray &server, IndexSet &indexSet,
                LocalArray &client, const std::string requestID);
        void requestTo(GlobalArray &server, IndexSet &indexSet, LocalArray &client,
                uint16_t opID, const std::string requestID);
        void requestFrom(GlobalArray &server, const unsigned numberOfVariables,
                const int64_t *matrix, const uint64_t *variableStart,
                const uint64_t *variableEnd, LocalArray &client,
                const std::string requestID);
        void requestTo(GlobalArray &server, const unsigned numberOfVariables,
                const int64_t *matrix, const uint64_t *variableStart,
                const uint64_t *variableEnd, LocalArray &client, uint16_t opID,
                const std::string requestID);

        /// @brief request data from/to local array
        void requestFrom(LocalArray &server, uint64_t serverProcID,
                IndexSet &indexSet, LocalArray &client,
                const std::string requestID);
        void requestTo(LocalArray &server, uint64_t serverProcID,
                IndexSet &indexSet, LocalArray &client, uint16_t opID,
                const std::string requestID);

        /// @brief send user-defined message to a given proc
        void exportUserDefined(uint64_t procID, std::string path);

        /// @brief perform the data exchange
        void exchange(bool *MatrixOfSendTo, const char *tag);

	/// @brief perform the data exchange with a single partner
	void exchange(uint64_t procID, const char *tag);

	/// @brief perfrom the data exchange with detected partners
	uint64_t exchange(const char *tag, bool stoppingScheduler);

        /// @brief remove "this." from path string
        std::string simplifyPath(std::string path);

        /// @brief create object
        void setObject(NameSpace *scope);

        /// @brief create object
        void setObject(std::string path, LocalArray *localArray);

        /// @brief create object
        void setObject(std::string path, GlobalArray *globalArray);

        /// @brief get object
        NamedObject *getObject(std::string path);

        /// @brief delete object
        void deleteObject(std::string path, bool collective = false);

        /// @brief clear path
        void clearPath(std::string path);

        /// @brief clear imported objects
        void clearImported();

        /// @brief clear imported objects
        void clearImported(uint64_t procID);

        /// @brief check whether object exists
        bool hasObject(std::string path);

        void setVerbose(bool _verbose);

        bool isVerbose() const;

        bool isFinalizing() const {return _finalizing;};

        void share(std::vector<std::string> nameList);

        void globalize(std::vector<std::string> nameList, const unsigned nGridDims, 
            const uint64_t gridDimSize[7], uint64_t procStart);

        void privatize(std::vector<std::string> nameList);

        /// @brief create 1d local array from a string
        void fromString(std::string string, std::string path);

        /// @brief create nd local array from a buffer
        LocalArray *fromBuffer(const char *buffer, const char kind, const int nDims, const ssize_t *dimSize, const ssize_t *strides, std::string path);

        /// @brief create string from 1d local array
        std::string toString(std::string path);

        /// @brief add worker
        void addWorker(uint64_t procID);

        /// @brief set scheduler for asynchronization
        void setScheduler(uint64_t boundOfDelay, uint64_t smallestBatch, uint64_t largestBatch);

        /// @brief unset scheduler and use default asynchronization
        void unsetScheduler();

        /// @brief enable scheduler
        void enableScheduler();

        /// @brief disable scheduler
        void disableScheduler();

    protected:
        /// @brief send data
        /// @param numberOfBytes the number of bytes of the data size
        /// @param dataBuffer the pointer to the data

        void send(size_t numberOfBytes, void *dataBuffer) {
            _nal.addDataBlockToSend(dataBuffer, numberOfBytes);
        }

        /// @brief receive data
        /// @param numberOfBytes the number of bytes of the data size
        /// @param dataBuffer the pointer to the data

        void receive(size_t numberOfBytes, void *dataBuffer) {
            _nal.addDataBlockToReceive(dataBuffer, numberOfBytes);
        }

        /// @brief exchange data
        /// @param partnerRank the rank of the partner to exchange data with

        void exchangeWith(uint64_t partnerRank) {
            _nal.exchangeDataBlocksWith(partnerRank);
        }

	/// @brief detect a single partner
	uint64_t detect(const char *tag);

        /// @brief get object by path
        NamedObject *getObjectByPath(std::string path);

        /// @brief get object path
        NameSpace *getObjectParent(std::string path, bool createItIfNotExists =
                false);

        /// @brief create imported object path
        NameSpace *createImportedObjectParent(uint64_t procID, std::string path);

        /// @brief create imported array
        LocalArray *createImportedArray(uint64_t procID, std::string path,
                uint64_t *arrayShapeSerialization);

        /// @brief serialize array shapes for export
        void serializeExport();

        /// @brief compute connectivity matrix
        bool* computeConnectivity(bool* MatrixOfSendTo);
    };
}

#endif /* BSPRUNTIME_HPP_ */
