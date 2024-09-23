// clang-format off(build/include_order)
#include <clang/AST/AST.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Lex/Lexer.h>
#include <clang/Rewrite/Core/Rewriter.h>
#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/Tooling.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Error.h>
// clang-format on

#include <fstream>
#include <string>
#include <vector>

#include "tachyon/base/files/file.h"
#include "tachyon/base/strings/string_number_conversions.h"

using namespace clang;
using namespace clang::tooling;
using namespace tachyon;

class FunctionAndGlobalExtractor
    : public RecursiveASTVisitor<FunctionAndGlobalExtractor> {
 public:
  FunctionAndGlobalExtractor(Rewriter& rewriter, const std::string& input_file,
                             const std::string& output_dir,
                             size_t functions_per_file,
                             size_t max_lines_per_file)
      : rewriter_(rewriter),
        input_file_(input_file),
        output_dir_(output_dir),
        functions_per_file_(functions_per_file),
        max_lines_per_file_(max_lines_per_file) {}

  bool VisitFunctionDecl(FunctionDecl* f) {
    SourceManager& sm = rewriter_.getSourceMgr();
    if (sm.isInMainFile(f->getLocation()) && f->hasBody()) {
      std::string func_decl = f->getReturnType().getAsString() + " " +
                              f->getNameInfo().getAsString() + "(";

      // Get function parameters
      for (unsigned i = 0; i < f->getNumParams(); ++i) {
        ParmVarDecl* param = f->getParamDecl(i);
        if (i > 0) func_decl += ", ";
        func_decl += param->getOriginalType().getAsString() + " " +
                     param->getNameAsString();
      }
      func_decl += ");";

      function_declarations_.push_back(func_decl);

      if (f->doesThisDeclarationHaveABody()) {
        // Get function definition
        SourceLocation start_loc = f->getSourceRange().getBegin();
        SourceLocation end_loc = f->getSourceRange().getEnd();

        // Extend EndLoc to include the entire function body
        end_loc =
            Lexer::getLocForEndOfToken(end_loc, 0, sm, rewriter_.getLangOpts());

        std::string func_def = getSourceText(sm, start_loc, end_loc);
        function_definitions_.push_back(func_def);
      }
    }
    return true;
  }

  bool VisitVarDecl(VarDecl* v) {
    SourceManager& sm = rewriter_.getSourceMgr();
    if (sm.isInMainFile(v->getLocation()) && v->isFileVarDecl() &&
        !v->isStaticLocal()) {
      // Get variable declaration
      QualType qt = v->getType();
      std::string var_type = v->getType().getAsString();
      std::string var_name = v->getNameAsString();
      std::string var_decl;

      if (qt->isArrayType()) {
        const ConstantArrayType* array_type =
            v->getASTContext().getAsConstantArrayType(qt);
        llvm::APInt array_size = array_type->getSize();
        std::string elem_type = array_type->getElementType().getAsString();
        var_decl = "extern " + elem_type + " " + var_name + "[" +
                   std::to_string(array_size.getLimitedValue()) + "];";
      } else {
        var_decl = "extern " + qt.getAsString() + " " + var_name + ";";
      }
      global_variable_declarations_.push_back(var_decl);

      // Get variable definition
      if (v->hasInit()) {
        SourceLocation start_loc = v->getSourceRange().getBegin();
        SourceLocation end_loc = v->getSourceRange().getEnd();
        end_loc =
            Lexer::getLocForEndOfToken(end_loc, 0, sm, rewriter_.getLangOpts());

        std::string var_def = getSourceText(sm, start_loc, end_loc) + ";";
        global_variable_definitions_.push_back(var_def);
      } else {
        std::string var_def = var_type + " " + var_name + ";";
        global_variable_definitions_.push_back(var_def);
      }
    }
    return true;
  }

  void WriteHeaderFile() {
    base::FilePath output_path =
        base::FilePath(output_dir_).Append("functions.h");
    base::File output_file(
        output_path, base::File::FLAG_CREATE_ALWAYS | base::File::FLAG_WRITE);
    const std::string_view header_guard =
        "#ifndef FUNCTIONS_H\n#define FUNCTIONS_H\n";
    output_file.WriteAtCurrentPos(header_guard.data(), header_guard.size());
    // Write global variable declarations
    for (const auto& var_decl : global_variable_declarations_) {
      output_file.WriteAtCurrentPos(var_decl.c_str(), var_decl.size());
      output_file.WriteAtCurrentPos("\n", 1);
    }
    // Write function declarations
    for (const auto& func_decl : function_declarations_) {
      output_file.WriteAtCurrentPos(func_decl.c_str(), func_decl.size());
      output_file.WriteAtCurrentPos("\n", 1);
    }
    const std::string_view header_guard_end = "\n#endif // FUNCTIONS_H\n";
    output_file.WriteAtCurrentPos(header_guard_end.data(),
                                  header_guard_end.size());
  }

  void WriteSourceFiles() {
    size_t file_count = 1;
    size_t total_definitions = function_definitions_.size();

    size_t function_index = 0;
    while (function_index < total_definitions) {
      base::FilePath output_path =
          base::FilePath(output_dir_)
              .Append("part_" + base::NumberToString(file_count) + ".cpp");
      base::File output_file(
          output_path, base::File::FLAG_CREATE_ALWAYS | base::File::FLAG_WRITE);
      size_t current_line_count = 0;

      // Write includes
      size_t includes_line_count =
          std::count(includes_code_.begin(), includes_code_.end(), '\n');
      output_file.WriteAtCurrentPos(includes_code_.c_str(),
                                    includes_code_.size());
      current_line_count += includes_line_count;

      // Include the header file
      const std::string_view include_header_code =
          "#include \"functions.h\"\n\n";
      output_file.WriteAtCurrentPos(include_header_code.data(),
                                    include_header_code.size());
      current_line_count += 2;

      // Include global variable definitions in the first file
      if (file_count == 1) {
        std::string global_vars_code;
        for (const auto& var_def : global_variable_definitions_) {
          output_file.WriteAtCurrentPos(var_def.c_str(), var_def.size());
          output_file.WriteAtCurrentPos("\n", 1);
          current_line_count +=
              std::count(var_def.begin(), var_def.end(), '\n') + 1;
        }
      }

      size_t functions_written = 0;
      while (function_index < total_definitions &&
             functions_written < functions_per_file_ &&
             current_line_count < max_lines_per_file_) {
        std::string func_def = function_definitions_[function_index];
        size_t func_def_line_count =
            std::count(func_def.begin(), func_def.end(), '\n');

        if (func_def_line_count + current_line_count > max_lines_per_file_ &&
            functions_written > 0) {
          break;
        }

        func_def += "\n\n";
        output_file.WriteAtCurrentPos(func_def.c_str(), func_def.size());
        current_line_count += func_def_line_count;
        ++function_index;
        ++functions_written;
      }

      ++file_count;
    }
  }

  void SetIncludesCode(const std::string& code) { includes_code_ = code; }

 private:
  Rewriter& rewriter_;
  std::string input_file_;
  std::string output_dir_;
  size_t functions_per_file_;
  size_t max_lines_per_file_;
  std::vector<std::string> function_declarations_;
  std::vector<std::string> function_definitions_;
  std::vector<std::string> global_variable_declarations_;
  std::vector<std::string> global_variable_definitions_;
  std::string includes_code_;

  std::string getSourceText(const SourceManager& sm, SourceLocation start,
                            SourceLocation end) {
    SourceRange range(start, end);
    bool invalid = false;
    const char* start_buf = sm.getCharacterData(start, &invalid);
    const char* end_buf = sm.getCharacterData(end, &invalid);
    if (invalid) return "";

    return std::string(start_buf, end_buf - start_buf);
  }
};

class FunctionAndGlobalExtractorASTConsumer : public ASTConsumer {
 public:
  FunctionAndGlobalExtractorASTConsumer(Rewriter& rewriter,
                                        const std::string& input_file,
                                        const std::string& output_dir,
                                        size_t functions_per_file,
                                        size_t max_lines_per_file)
      : visitor_(rewriter, input_file, output_dir, functions_per_file,
                 max_lines_per_file) {}

  void HandleTranslationUnit(ASTContext& context) override {
    visitor_.TraverseDecl(context.getTranslationUnitDecl());
    visitor_.WriteHeaderFile();
    visitor_.WriteSourceFiles();
  }

  void SetIncludesCode(const std::string& code) {
    visitor_.SetIncludesCode(code);
  }

 private:
  FunctionAndGlobalExtractor visitor_;
};

class FunctionAndGlobalExtractorFrontendAction : public ASTFrontendAction {
 public:
  FunctionAndGlobalExtractorFrontendAction(const std::string& input_file,
                                           const std::string& output_dir,
                                           size_t functions_per_file,
                                           size_t max_lines_per_file)
      : input_file_(input_file),
        output_dir_(output_dir),
        functions_per_file_(functions_per_file),
        max_lines_per_file_(max_lines_per_file) {}

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance& CI,
                                                 StringRef file) override {
    rewriter_.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());

    // Read includes and macros from the beginning of the file
    std::string includes_code = readIncludesAndMacros(input_file_);

    auto ast_consumer = std::make_unique<FunctionAndGlobalExtractorASTConsumer>(
        rewriter_, input_file_, output_dir_, functions_per_file_,
        max_lines_per_file_);
    ast_consumer->SetIncludesCode(includes_code);
    return ast_consumer;
  }

 private:
  Rewriter rewriter_;
  std::string input_file_;
  std::string output_dir_;
  size_t functions_per_file_;
  size_t max_lines_per_file_;

  std::string readIncludesAndMacros(const std::string& file_path) {
    std::ifstream input_stream(file_path);
    std::string line;
    std::string includes;
    while (std::getline(input_stream, line)) {
      std::string trimmed_line = line;
      trimmed_line.erase(0, trimmed_line.find_first_not_of(" \t"));
      if (trimmed_line.rfind("#include", 0) == 0 ||
          trimmed_line.rfind("#define", 0) == 0) {
        includes += line + "\n";
      } else if (trimmed_line.empty()) {
        continue;
      } else {
        break;
      }
    }
    includes += "\n";
    return includes;
  }
};

class MyFrontendActionFactory : public clang::tooling::FrontendActionFactory {
 public:
  MyFrontendActionFactory(const std::string& input_file,
                          const std::string& output_dir,
                          size_t functions_per_file, size_t max_lines_per_file)
      : input_file_(input_file),
        output_dir_(output_dir),
        functions_per_file_(functions_per_file),
        max_lines_per_file_(max_lines_per_file) {}

  std::unique_ptr<clang::FrontendAction> create() override {
    return std::make_unique<FunctionAndGlobalExtractorFrontendAction>(
        input_file_, output_dir_, functions_per_file_, max_lines_per_file_);
  }

 private:
  std::string input_file_;
  std::string output_dir_;
  size_t functions_per_file_;
  size_t max_lines_per_file_;
};

int main(int argc, const char** argv) {
  // Define command-line options category
  llvm::cl::OptionCategory tool_category("split-functions-tool options");

  llvm::cl::opt<std::string> output_dir_option(
      "output_dir", llvm::cl::desc("Specify output directory"),
      llvm::cl::value_desc("directory"), llvm::cl::init("split_files"));

  // Parse command-line options
  auto expected_parser = CommonOptionsParser::create(argc, argv, tool_category);
  if (!expected_parser) {
    llvm::errs() << expected_parser.takeError();
    return 1;
  }
  CommonOptionsParser& options_parser = expected_parser.get();

  // Create ClangTool
  ClangTool tool(options_parser.getCompilations(),
                 options_parser.getSourcePathList());

  // Get source files
  const auto& sources = options_parser.getSourcePathList();
  if (sources.empty()) {
    llvm::errs() << "No source files provided.\n";
    return 1;
  }

  // Initialize variables
  std::string input_file = sources[0];
  std::string output_dir = output_dir_option;
  size_t functions_per_file = 1000;
  size_t max_lines_per_file = 100000;

  // Create a custom FrontendActionFactory
  auto action_factory = std::make_unique<MyFrontendActionFactory>(
      input_file, output_dir, functions_per_file, max_lines_per_file);

  // Run the tool
  int result = tool.run(action_factory.get());

  return result;
}
