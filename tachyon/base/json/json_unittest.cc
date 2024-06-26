#include "tachyon/base/json/json.h"

#include <utility>
#include <vector>

#include "gtest/gtest.h"

namespace tachyon::base {

namespace {

struct SimpleData {
  std::string message;
  int index = 0;
  bool flag = false;
  std::vector<unsigned int> data;

  bool operator==(const SimpleData& other) const {
    return message == other.message && index == other.index &&
           flag == other.flag && data == other.data;
  }
  bool operator!=(const SimpleData& other) const { return !operator==(other); }
};

class JsonTest : public testing::Test {
 public:
  void SetUp() override {
    expected_simple_data_.message = "hello world";
    expected_simple_data_.index = 1;
    expected_simple_data_.flag = true;
    expected_simple_data_.data = std::vector<unsigned int>{0, 2, 4};
  }

 protected:
  SimpleData expected_simple_data_;
};

}  // namespace

template <>
class RapidJsonValueConverter<SimpleData> {
 public:
  template <typename Allocator>
  static rapidjson::Value From(const SimpleData& value, Allocator& allocator) {
    rapidjson::Value object(rapidjson::kObjectType);
    // NOTE: avoid unnecessary copy.
    std::string_view message = value.message;
    AddJsonElement(object, "message", message, allocator);
    AddJsonElement(object, "index", value.index, allocator);
    AddJsonElement(object, "flag", value.flag, allocator);
    AddJsonElement(object, "data", value.data, allocator);
    return object;
  }

  static bool To(const rapidjson::Value& json_value, std::string_view key,
                 SimpleData* value, std::string* error) {
    std::string message;
    int index;
    bool flag;
    std::vector<unsigned int> data;
    if (!ParseJsonElement(json_value, "message", &message, error)) return false;
    if (!ParseJsonElement(json_value, "index", &index, error)) return false;
    if (!ParseJsonElement(json_value, "flag", &flag, error)) return false;
    if (!ParseJsonElement(json_value, "data", &data, error)) return false;

    value->message = std::move(message);
    value->index = index;
    value->flag = flag;
    value->data = std::move(data);
    return true;
  }
};

TEST_F(JsonTest, LoadAndParseJson) {
  SimpleData simple_data;
  std::string error;
  ASSERT_TRUE(
      LoadAndParseJson(FilePath("tachyon/base/json/test/simple_data.json"),
                       &simple_data, &error));
  EXPECT_TRUE(error.empty());

  EXPECT_EQ(simple_data, expected_simple_data_);
}

TEST_F(JsonTest, ParseInvalidJson) {
  // missing key
  std::string json = R"({})";
  SimpleData simple_data;
  std::string error;
  ASSERT_FALSE(ParseJson(json, &simple_data, &error));
  EXPECT_EQ(error, "\"message\" key is not found");

  // invalid value
  json = R"({"message":3})";
  ASSERT_FALSE(ParseJson(json, &simple_data, &error));
  EXPECT_EQ(error,
            "\"message\" expects type \"string\" but type \"number\" comes");
}

TEST_F(JsonTest, WriteToJson) {
  std::string json = WriteToJson(expected_simple_data_);
  EXPECT_EQ(
      json,
      R"({"message":"hello world","index":1,"flag":true,"data":[0,2,4]})");
}

}  // namespace tachyon::base
