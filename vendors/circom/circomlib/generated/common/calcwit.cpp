#include <iomanip>
#include <sstream>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <vector>
#include <chrono>
#include "calcwit.hpp"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

extern void run(Circom_CalcWit* ctx);

std::string int_to_hex( u64 i )
{
  std::stringstream stream;
  stream << "0x"
         << std::setfill ('0') << std::setw(16)
         << std::hex << i;
  return stream.str();
}

u64 fnv1a(std::string s) {
  u64 hash = 0xCBF29CE484222325LL;
  for(char& c : s) {
    hash ^= u64(c);
    hash *= 0x100000001B3LL;
  }
  return hash;
}

Circom_CalcWit::Circom_CalcWit (Circom_Circuit *aCircuit, uint maxTh) {
  circuit = aCircuit;
  inputSignalAssignedCounter = get_main_input_signal_no();
  inputSignalAssigned = new bool[inputSignalAssignedCounter];
  for (uint i = 0; i< inputSignalAssignedCounter; i++) {
    inputSignalAssigned[i] = false;
  }
  signalValues = new FrElement[get_total_signal_no()];
  Fr_str2element(&signalValues[0], "1", 10);
  componentMemory = new Circom_Component[get_number_of_components()];
  circuitConstants = circuit ->circuitConstants;
  templateInsId2IOSignalInfo = circuit -> templateInsId2IOSignalInfo;

  maxThread = maxTh;

  // parallelism
  numThread = 0;

}

Circom_CalcWit::~Circom_CalcWit() {
  // ...
}

uint Circom_CalcWit::getInputSignalHashPosition(u64 h) {
  uint n = get_size_of_input_hashmap();
  uint pos = (uint)(h % (u64)n);
  if (circuit->InputHashMap[pos].hash!=h){
    uint inipos = pos;
    pos++;
    while (pos != inipos) {
      if (circuit->InputHashMap[pos].hash==h) return pos;
      if (circuit->InputHashMap[pos].hash==0) {
	fprintf(stderr, "Signal not found\n");
	assert(false);
      }
      pos = (pos+1)%n;
    }
    fprintf(stderr, "Signals not found\n");
    assert(false);
  }
  return pos;
}

void Circom_CalcWit::tryRunCircuit(){
  if (inputSignalAssignedCounter == 0) {
    run(this);
  }
}

void Circom_CalcWit::setInputSignal(u64 h, uint i,  FrElement const & val){
  if (inputSignalAssignedCounter == 0) {
    fprintf(stderr, "No more signals to be assigned\n");
    assert(false);
  }
  uint pos = getInputSignalHashPosition(h);
  if (i >= circuit->InputHashMap[pos].signalsize) {
    fprintf(stderr, "Input signal array access exceeds the size\n");
    assert(false);
  }

  uint si = circuit->InputHashMap[pos].signalid+i;
  if (inputSignalAssigned[si-get_main_input_signal_start()]) {
    fprintf(stderr, "Signal assigned twice: %d\n", si);
    assert(false);
  }
  signalValues[si] = val;
  inputSignalAssigned[si-get_main_input_signal_start()] = true;
  inputSignalAssignedCounter--;
  tryRunCircuit();
}

u64 Circom_CalcWit::getInputSignalSize(u64 h) {
  uint pos = getInputSignalHashPosition(h);
  return circuit->InputHashMap[pos].signalsize;
}

std::string Circom_CalcWit::getTrace(u64 id_cmp){
  if (id_cmp == 0) return componentMemory[id_cmp].componentName;
  else{
    u64 id_father = componentMemory[id_cmp].idFather;
    std::string my_name = componentMemory[id_cmp].componentName;

    return Circom_CalcWit::getTrace(id_father) + "." + my_name;
  }


}

std::string Circom_CalcWit::generate_position_array(uint* dimensions, uint size_dimensions, uint index){
  std::string positions = "";

  for (uint i = 0 ; i < size_dimensions; i++){
    uint last_pos = index % dimensions[size_dimensions -1 - i];
    index = index / dimensions[size_dimensions -1 - i];
    std::string new_pos = "[" + std::to_string(last_pos) + "]";
    positions =  new_pos + positions;
  }
  return positions;
}

#define handle_error(msg) \
           do { perror(msg); exit(EXIT_FAILURE); } while (0)

Circom_Circuit* loadCircuit(std::string const &datFileName) {
  Circom_Circuit *circuit = new Circom_Circuit;

  int fd;
  struct stat sb;

  fd = open(datFileName.c_str(), O_RDONLY);
  if (fd == -1) {
    std::cout << ".dat file not found: " << datFileName << "\n";
    throw std::system_error(errno, std::generic_category(), "open");
  }

  if (fstat(fd, &sb) == -1) {          /* To obtain file size */
    throw std::system_error(errno, std::generic_category(), "fstat");
  }

  u8* bdata = (u8*)mmap(NULL, sb.st_size, PROT_READ , MAP_PRIVATE, fd, 0);
  close(fd);

  circuit->InputHashMap = new HashSignalInfo[get_size_of_input_hashmap()];
  uint dsize = get_size_of_input_hashmap()*sizeof(HashSignalInfo);
  memcpy((void *)(circuit->InputHashMap), (void *)bdata, dsize);

  circuit->witness2SignalList = new u64[get_size_of_witness()];
  uint inisize = dsize;
  dsize = get_size_of_witness()*sizeof(u64);
  memcpy((void *)(circuit->witness2SignalList), (void *)(bdata+inisize), dsize);

  circuit->circuitConstants = new FrElement[get_size_of_constants()];
  if (get_size_of_constants()>0) {
    inisize += dsize;
    dsize = get_size_of_constants()*sizeof(FrElement);
    memcpy((void *)(circuit->circuitConstants), (void *)(bdata+inisize), dsize);
  }

  std::map<u32,IODefPair> templateInsId2IOSignalInfo1;
  if (get_size_of_io_map()>0) {
    u32 index[get_size_of_io_map()];
    inisize += dsize;
    dsize = get_size_of_io_map()*sizeof(u32);
    memcpy((void *)index, (void *)(bdata+inisize), dsize);
    inisize += dsize;
    assert(inisize % sizeof(u32) == 0);
    assert(sb.st_size % sizeof(u32) == 0);
    u32 dataiomap[(sb.st_size-inisize)/sizeof(u32)];
    memcpy((void *)dataiomap, (void *)(bdata+inisize), sb.st_size-inisize);
    u32* pu32 = dataiomap;

    for (uint i = 0; i < get_size_of_io_map(); i++) {
      u32 n = *pu32;
      IODefPair p;
      p.len = n;
      IODef defs[n];
      pu32 += 1;
      for (u32 j = 0; j <n; j++){
        defs[j].offset=*pu32;
        u32 len = *(pu32+1);
        defs[j].len = len;
        defs[j].lengths = new u32[len];
        memcpy((void *)defs[j].lengths,(void *)(pu32+2),len*sizeof(u32));
        pu32 += len + 2;
      }
      p.defs = (IODef*)calloc(10, sizeof(IODef));
      for (u32 j = 0; j < p.len; j++){
        p.defs[j] = defs[j];
      }
	    templateInsId2IOSignalInfo1[index[i]] = p;
    }
  }
  circuit->templateInsId2IOSignalInfo = std::move(templateInsId2IOSignalInfo1);

  munmap(bdata, sb.st_size);

  return circuit;
}

bool check_valid_number(std::string & s, uint base){
  bool is_valid = true;
  if (base == 16){
    for (uint i = 0; i < s.size(); i++){
      is_valid &= (
        ('0' <= s[i] && s[i] <= '9') ||
        ('a' <= s[i] && s[i] <= 'f') ||
        ('A' <= s[i] && s[i] <= 'F')
      );
    }
  } else{
    for (uint i = 0; i < s.size(); i++){
      is_valid &= ('0' <= s[i] && s[i] < char(int('0') + base));
    }
  }
  return is_valid;
}

void json2FrElements (json val, std::vector<FrElement> & vval){
  if (!val.is_array()) {
    FrElement v;
    std::string s_aux, s;
    uint base;
    if (val.is_string()) {
      s_aux = val.get<std::string>();
      std::string possible_prefix = s_aux.substr(0, 2);
      if (possible_prefix == "0b" || possible_prefix == "0B"){
        s = s_aux.substr(2, s_aux.size() - 2);
        base = 2;
      } else if (possible_prefix == "0o" || possible_prefix == "0O"){
        s = s_aux.substr(2, s_aux.size() - 2);
        base = 8;
      } else if (possible_prefix == "0x" || possible_prefix == "0X"){
        s = s_aux.substr(2, s_aux.size() - 2);
        base = 16;
      } else{
        s = s_aux;
        base = 10;
      }
      if (!check_valid_number(s, base)){
        std::ostringstream errStrStream;
        errStrStream << "Invalid number in JSON input: " << s_aux << "\n";
	      throw std::runtime_error(errStrStream.str() );
      }
    } else if (val.is_number()) {
        double vd = val.get<double>();
        std::stringstream stream;
        stream << std::fixed << std::setprecision(0) << vd;
        s = stream.str();
        base = 10;
    } else {
        std::ostringstream errStrStream;
        errStrStream << "Invalid JSON type\n";
	      throw std::runtime_error(errStrStream.str() );
    }
    Fr_str2element (&v, s.c_str(), base);
    vval.push_back(v);
  } else {
    for (uint i = 0; i < val.size(); i++) {
      json2FrElements (val[i], vval);
    }
  }
}

void loadJson(Circom_CalcWit *ctx, std::string filename) {
  std::ifstream inStream(filename);
  json j;
  inStream >> j;

  u64 nItems = j.size();
  // printf("Items : %llu\n",nItems);
  if (nItems == 0){
    ctx->tryRunCircuit();
  }
  for (json::iterator it = j.begin(); it != j.end(); ++it) {
    // std::cout << it.key() << " => " << it.value() << '\n';
    u64 h = fnv1a(it.key());
    std::vector<FrElement> v;
    json2FrElements(it.value(),v);
    uint signalSize = ctx->getInputSignalSize(h);
    if (v.size() < signalSize) {
	std::ostringstream errStrStream;
	errStrStream << "Error loading signal " << it.key() << ": Not enough values\n";
	throw std::runtime_error(errStrStream.str() );
    }
    if (v.size() > signalSize) {
	std::ostringstream errStrStream;
	errStrStream << "Error loading signal " << it.key() << ": Too many values\n";
	throw std::runtime_error(errStrStream.str() );
    }
    for (uint i = 0; i<v.size(); i++){
      try {
	// std::cout << it.key() << "," << i << " => " << Fr_element2str(&(v[i])) << '\n';
	ctx->setInputSignal(h,i,v[i]);
      } catch (std::runtime_error e) {
	std::ostringstream errStrStream;
	errStrStream << "Error setting signal: " << it.key() << "\n" << e.what();
	throw std::runtime_error(errStrStream.str() );
      }
    }
  }
}

void loadWitness(Circom_CalcWit *ctx, const absl::flat_hash_map<std::string, std::vector<FrElement>>& witness) {
  size_t nItems = witness.size();
  // printf("Items : %llu\n",nItems);
  if (nItems == 0){
    ctx->tryRunCircuit();
  }
  for (const auto& [key, value] : witness) {
    u64 h = fnv1a(key);
    uint signalSize = ctx->getInputSignalSize(h);
    if (value.size() < signalSize) {
	    std::ostringstream errStrStream;
	    errStrStream << "Error loading signal " << key << ": Not enough values\n";
	    throw std::runtime_error(errStrStream.str() );
    }
    if (value.size() > signalSize) {
	    std::ostringstream errStrStream;
	    errStrStream << "Error loading signal " << key << ": Too many values\n";
	    throw std::runtime_error(errStrStream.str() );
    }
    for (uint i = 0; i<value.size(); i++){
      try {
	      // std::cout << key << "," << i << " => " << Fr_element2str(&(value[i])) << '\n';
	      ctx->setInputSignal(h,i,value[i]);
      } catch (std::runtime_error e) {
	      std::ostringstream errStrStream;
	      errStrStream << "Error setting signal: " << key << "\n" << e.what();
	      throw std::runtime_error(errStrStream.str() );
      }
    }
  }
  if (ctx->getRemaingInputsToBeSet()!=0) {
    std::cerr << "Not all inputs have been set. Only " << get_main_input_signal_no()-ctx->getRemaingInputsToBeSet() << " out of " << get_main_input_signal_no() << std::endl;
    assert(false);
  }
}

void writeBinWitness(Circom_CalcWit *ctx, std::string const &wtnsFileName) {
  FILE *write_ptr;

  write_ptr = fopen(wtnsFileName.c_str(),"wb");

  fwrite("wtns", 4, 1, write_ptr);

  u32 version = 2;
  fwrite(&version, 4, 1, write_ptr);

  u32 nSections = 2;
  fwrite(&nSections, 4, 1, write_ptr);

  // Header
  u32 idSection1 = 1;
  fwrite(&idSection1, 4, 1, write_ptr);

  u32 n8 = Fr_N64*8;

  u64 idSection1length = 8 + n8;
  fwrite(&idSection1length, 8, 1, write_ptr);

  fwrite(&n8, 4, 1, write_ptr);

  fwrite(Fr_q.longVal, Fr_N64*8, 1, write_ptr);

  uint Nwtns = get_size_of_witness();

  u32 nVars = (u32)Nwtns;
  fwrite(&nVars, 4, 1, write_ptr);

  // Data
  u32 idSection2 = 2;
  fwrite(&idSection2, 4, 1, write_ptr);

  u64 idSection2length = (u64)n8*(u64)Nwtns;
  fwrite(&idSection2length, 8, 1, write_ptr);

  FrElement v;

  for (int i=0;i<Nwtns;i++) {
    ctx->getWitness(i, &v);
    Fr_toLongNormal(&v, &v);
    fwrite(v.longVal, Fr_N64*8, 1, write_ptr);
  }
  fclose(write_ptr);
}
