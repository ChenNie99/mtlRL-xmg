#include <iostream>
#include <string>
#include <cstring>
#include <fstream>
#include <stack>
#include <algorithm>

#include <pybind11/pybind11.h>
namespace py = pybind11;


using namespace std;

typedef struct Node {
    string name;
    bool leaf = false;       // false: intermediate nodes; true: leaf/root nodes
    bool inv = false;       // false: origin; true: need an inverter
}node;

void CheckFile(bool iFile, const string& filename)
{
    if (!iFile) {
        cerr << "Error: Cannot open file " << filename << "!" << endl;
        exit(2);
    }
}

void hex2bin(string& line) {
    size_t position_1h = line.find("1'h");
    while (position_1h != string::npos) {
        line[position_1h + 2] = 'b';
        position_1h = line.find("1'h", position_1h + 3);
    }
}

void RemoveBracket(string &expr) {
    string temp;
    for (int i = 0; i < expr.length(); ++i) {
        if (expr[i] == '(') {
            string buf;
            i++;
            for (;i < expr.length(); ++i) {
                buf += expr[i];
                if (expr[i] == '&' || expr[i] == '|' || expr[i] == '^') {
                    temp += ('(' + buf.substr(0, buf.length()));
                    break;
                } else if (expr[i] == ')') {
                    temp += buf.substr(0, buf.length() - 1);
                    break;
                } else if (expr[i] == '(') {
                    temp += ('(' + buf.substr(0, buf.length() - 1));
                    i--;
                    break;
                }
            }
            continue;
        }
        temp += expr[i];
    }
    expr = temp;
}

int CountOperands(const string& rhs) {
    int count = 0;
    for (char i : rhs) {
        if (i == '&' || i == '|' || i == '^') {
            count++;
        }
    }
    return count;
}

void v2v(const string& input_file = "") {
    string filename;
    if (input_file.empty()) {
        cout << "Input Verilog filename: ";
        cin >> filename;
    } else {
        filename = input_file;
    }

    string output = filename.substr(0, filename.length() - 2) + "_out.v";

    ifstream iVerilog;
    iVerilog.open(filename);
    CheckFile((bool) iVerilog, filename);
    string line;

    ofstream oVerilog;
    oVerilog.open(output);
    ofstream assignTemp;
    assignTemp.open("temporary.v");

    int temp_num = 0;

    while (getline(iVerilog, line)) {
        if (line.length() > 2 && (line.find("(*") != std::string::npos || line.find("/*") != std::string::npos)) {
            continue;
        }

        hex2bin(line);

        if (line.find("input") != std::string::npos || line.find("output") != std::string::npos ||
                   line.find("endmodule") != std::string::npos) {
            RemoveBracket(line);
            assignTemp << line << '\n';
        } else if (line.find("assign") != std::string::npos) {
            auto partition = line.find_first_of('=');
            string lhs = line.substr(0, partition - 1);
            string rhs = line.substr(partition);
            int num_operands = CountOperands(rhs);
            if (rhs.find('?') != std::string::npos && rhs.find(':') != std::string::npos) {
                string op1, op2, op3;
                int temp_pt = 0;
                while (rhs[temp_pt] == '=' || rhs[temp_pt] == ' ') {
                    temp_pt++;
                }
                while (rhs[temp_pt] != ' ' && rhs[temp_pt] != '?') {
                    op1 += rhs[temp_pt];
                    temp_pt++;
                }

                while (rhs[temp_pt] == '?' || rhs[temp_pt] == ' ') {
                    temp_pt++;
                }
                while (rhs[temp_pt] != ' ' && rhs[temp_pt] != ':') {
                    op2 += rhs[temp_pt];
                    temp_pt++;
                }

                while (rhs[temp_pt] == ':' || rhs[temp_pt] == ' ') {
                    temp_pt++;
                }
                while (rhs[temp_pt] != ' ' && rhs[temp_pt] != ';') {
                    op3 += rhs[temp_pt];
                    temp_pt++;
                }
                string temp_expr1 = "_temp" + to_string(temp_num) + "_";
                oVerilog << "  wire " << temp_expr1 << ";\n";
                assignTemp << "  assign " << temp_expr1 << " = " << op1 << " & " << op2 << ";\n";
                temp_num++;
                string temp_expr2 = "_temp" + to_string(temp_num) + "_";
                oVerilog << "  wire " << temp_expr2 << ";\n";
                assignTemp << "  assign " << temp_expr2 << " = ~" << op1 << " & " << op3 << ";\n";
                temp_num++;

                assignTemp << lhs << " = " << temp_expr1 << " | " << temp_expr2 << ";\n";
            } else if (num_operands >= 1) {
                stack<char> rhs_stack;
                for (int i = 0; i < rhs.length(); ++i) {
                    if (rhs[i] == ')'/* && rhs[i + 1] != ';'*/) {
                        string temp_str;
                        while (rhs_stack.top() != '(') {
                            temp_str += rhs_stack.top();
                            rhs_stack.pop();
                        }
                        // reach the last '('
                        rhs_stack.pop();
                        // reverse the order of temp_str
                        reverse(temp_str.begin(), temp_str.end());
                        string temp_wire = "_temp" + to_string(temp_num) + "_";
                        oVerilog << "  wire " << temp_wire << ";\n";
                        assignTemp << "  assign " << temp_wire << " = " << temp_str << ";\n";
                        for (auto j:temp_wire) {
                            rhs_stack.push(j);
                        }
                        temp_num++;
                    } else {
                        rhs_stack.push(rhs[i]);
                    }
                }
                string temp_str;
                while (!rhs_stack.empty()) {
                    temp_str += rhs_stack.top();
                    rhs_stack.pop();
                }
                reverse(temp_str.begin(), temp_str.end());
                assignTemp << lhs << ' ' << temp_str << "\n";
            } else {
                RemoveBracket(line);
                assignTemp << line << '\n';
            }
        } else if (!line.empty()) {
            oVerilog << line << '\n';
        }
    }

    iVerilog.close();
    oVerilog.close();
    assignTemp.close();

    ifstream iFile;
    iFile.open("temporary.v");
    ofstream oFile;
    oFile.open(output, ios_base::app);

    while (getline(iFile, line)) {
        oFile << line << '\n';
    }

    iFile.close();
    oFile.close();
}

void CheckOperand(node& a) {
    if (a.name[0] == '~') {
        a.inv = true;
        a.name = a.name.substr(1);
    }
    if (a.name[0] != 'n') {
        a.leaf = true;
    }
}

void PrintLine(ofstream& oVerilog, node& res, node& dest, int& temp_num) {
    if (res.name == "0" || res.name == "1") {
        oVerilog << "  temp" << temp_num << " [label=\""
                 << (res.name.back() == '1' ^ res.inv ? '1' : '0')
                 << "\", shape=none];\n"
                    "  temp" << temp_num << " -> " << dest.name << ";\n";
        ++temp_num;
    } else {
        /*if (res.leaf) {
            oVerilog << "  " << res.name << " [shape=none];\n";
        }*/
        oVerilog << "  " << res.name << " -> " << dest.name;
        if (res.inv) {
            oVerilog << " [arrowhead=odot]";
        }
        oVerilog << ";\n";
    }
}

void v2dot(bool xor3, const string& input_file = "") {
    string filename;
    if (input_file.empty()) {
        cout << "Input Verilog filename: ";
        cin >> filename;
    } else {
        filename = input_file;
    }
    string output = filename.substr(0, filename.length() - 2) + ".dot";
    string temp_out = filename.substr(0, filename.length() - 2) + "_temp.txt";

    ifstream iVerilog;
    iVerilog.open(filename);
    CheckFile((bool) iVerilog, filename);
    string line;

    ofstream oVerilog;
    oVerilog.open(output);

    ofstream oTemp;
    oTemp.open(temp_out);
    oTemp << "                              \n";


    std::string digraph_name = filename.substr(0, filename.length() - 2);
    for (char & letter : digraph_name) {
        if (letter == '/' || letter == '.') {
            letter = '_';
        }
    }
    oVerilog << "digraph " << digraph_name << " {\n  node[shape=circle];\n";

    // set the dot graph format of X and Y
    if (getline(iVerilog, line)) {      // read the first line
        int mark = line.find_first_of('(');
        mark++;
        while (mark+1 < line.length() && line[mark+1] != ')' && line[mark+1] != ';') {
            while (mark < line.length() && (line[mark] == ' ' || line[mark] == ',' || line[mark] == '(')) {
                mark++;
            }
            string temp_name;
            while (line[mark] != ' ' && line[mark] != ',' && line[mark] != ')') {
                temp_name+=line[mark];
                mark++;
            }
            if (temp_name[0] == 'x') {
                oVerilog << "  " << temp_name << " [shape=invhouse, style=filled, fillcolor=orange];\n";
            } else {
                oVerilog << "  " << temp_name << " [shape=house, style=filled, fillcolor=green];\n";
            }
        }
    }

    int temp_num = 0;
    int x_num = 0;
    int y_num = 0;
    int n_shift = 0;
    int n_id = 0;

    while (getline(iVerilog, line)) {
        if (line.find("output") != std::string::npos) {
            int pos = line.find_last_of('y');
            pos++;
            string num_string;
            while (line[pos] != ' ' && line[pos] != ';') {
                num_string += line[pos];
                pos++;
            }
            y_num = atoi(num_string.c_str()) + 1;
            continue;
        } else if (line.find("input") != std::string::npos) {
            int pos = line.find_last_of('x');
            pos++;
            string num_string;
            while (line[pos] != ' ' && line[pos] != ';') {
                num_string += line[pos];
                pos++;
            }
            x_num = atoi(num_string.c_str()) + 1;
            continue;
        } else if (line.find("wire") != std::string::npos) {
            int pos = line.find_first_of('n');
            pos++;
            string num_string;
            while (line[pos] != ' ' && line[pos] != ',') {
                num_string += line[pos];
                pos++;
            }
            n_shift = atoi(num_string.c_str());
            pos = line.find_last_of('n');
            pos++;
            num_string = "";
            while (line[pos] != ' ' && line[pos] != ';') {
                num_string += line[pos];
                pos++;
            }
            n_id = atoi(num_string.c_str()) + 1;
            continue;
        }
        if (line.find("assign") == std::string::npos) continue; // continue to the next loop
        auto partition = line.find_first_of('=');
        string lhs = line.substr(0, partition - 1);
        string rhs = line.substr(partition+1);
        node operand[3];
        node dest;
        int num_operands = CountOperands(rhs);
        // case 1: assign a = (~)b
        if (num_operands == 0) {

            int temp = lhs.find_first_of("assign");
            temp += 7;
            while (temp < lhs.length() && lhs[temp] != ' ' && lhs[temp] != '=') {
                dest.name += lhs[temp];
                temp++;
            }
            /*if (dest.name[0] != 'n') {
                oVerilog << "  " << dest.name << " [shape=none];\n";
            }*/

            temp = rhs.find_first_not_of(' ');
            while (rhs[temp] != ' ') {
                operand[0].name += rhs[temp];
                temp++;
            }
            CheckOperand(operand[0]);

            // rhs is a constant
            if (operand[0].name.find("1'b") != std::string::npos) {
                operand[0].name = operand[0].name.substr(3);
                PrintLine(oVerilog, operand[0], dest, temp_num);
                oTemp << dest.name << ' '
                      << (operand[0].name.back() == '1' ^ operand[0].inv ? 'T' : 'F')
                      << ";\n";
                continue;
            }

            PrintLine(oVerilog, operand[0], dest, temp_num);
            oTemp << dest.name << " ";
            if (operand[0].inv) {
                oTemp << "~";
            }
            oTemp << operand[0].name << ";\n";
            continue;

        }
        // case 2: 2-input gate
        else if (num_operands == 1) {
            string node_shape = "circle";
            if (line.find('&') != std::string::npos) {
                node_shape = "pentagon, regular=true";
                oTemp << "&";
            } else if (line.find('^') != std::string::npos) {
                node_shape = "octagon, regular=true";
                oTemp << "^";
            } else if (line.find('|') != std::string::npos) {
                node_shape = "pentagon, regular=true, orientation=180";
                oTemp << "|";
            }
            int temp = lhs.find_first_of("assign");
            temp += 7;
            while (temp < lhs.length() && lhs[temp] != ' ' && lhs[temp] != '=') {
                dest.name += lhs[temp];
                temp++;
            }

            temp = rhs.find_first_not_of(' ');
            while (rhs[temp] != ' ') {
                operand[0].name += rhs[temp];
                temp++;
            }
            CheckOperand(operand[0]);

            temp += 3;
            while (rhs[temp] != ' ' && rhs[temp] != ';') {
                operand[1].name += rhs[temp];
                temp++;
            }
            CheckOperand(operand[1]);

            oTemp << dest.name << " ";
            if (operand[0].inv) {
                oTemp << "~";
            }
            oTemp << operand[0].name << " ";
            if (operand[1].inv) {
                oTemp << "~";
            }
            oTemp << operand[1].name << " \n";

            oVerilog << "  " << dest.name << " [shape=" << node_shape << "];\n";
            PrintLine(oVerilog, operand[0], dest, temp_num);
            PrintLine(oVerilog, operand[1], dest, temp_num);
            continue;

        }
        // case 3: 3 input XOR gate
        else if (num_operands == 2 && line.find('^') != std::string::npos) {
            int temp = lhs.find_first_of("assign");
            temp += 7;
            while (temp < lhs.length() && lhs[temp] != ' ' && lhs[temp] != '=') {
                dest.name += lhs[temp];
                temp++;
            }

            temp = rhs.find_first_not_of(' ');
            while (rhs[temp] != ' ') {
                operand[0].name += rhs[temp];
                temp++;
            }
            CheckOperand(operand[0]);

            temp += 3;
            while (rhs[temp] != ' ' && rhs[temp] != ';') {
                operand[1].name += rhs[temp];
                temp++;
            }
            CheckOperand(operand[1]);

            temp += 3;
            while (rhs[temp] != ' ' && rhs[temp] != ';') {
                operand[2].name += rhs[temp];
                temp++;
            }

            // the third one is a constant
            if (operand[2].name.find("1'b") != std::string::npos) {
                if (operand[2].name.back() == '1') {
                    operand[0].inv = !operand[0].inv;
                    operand[1].inv = !operand[1].inv;
                }
                oTemp << "^" << dest.name << " ";
                if (operand[0].inv) {
                    oTemp << "~";
                }
                oTemp << operand[0].name << " ";
                if (operand[1].inv) {
                    oTemp << "~";
                }
                oTemp << operand[1].name << " \n";

                oVerilog << "  " << dest.name << " [shape=octagon, regular=true];\n";
                PrintLine(oVerilog, operand[0], dest, temp_num);
                PrintLine(oVerilog, operand[1], dest, temp_num);
                continue;
            } else {
                // XOR3 is allowed
                if (xor3) {
                    oVerilog << "  " << dest.name << " [shape=octagon, regular=true];\n";
                    for (auto i:operand) {
                        PrintLine(oVerilog, i, dest, temp_num);
                    }
                    oTemp << "@" << dest.name << " ";
                    for (const auto& i:operand) {
                        if (i.name == "0") {
                            oTemp << "F ";
                        } else if (i.name == "1") {
                            oTemp << "T ";
                        } else {
                            if (i.inv) {
                                oTemp << "~";
                            }
                            oTemp << i.name << " ";
                        }
                    }
                    oTemp << "\n";
                } else {
                    // first part
                    node temp_node;
                    temp_node.name = "n" + to_string(n_id);
                    n_id++;
                    oTemp << "^" << temp_node.name << " ";
                    if (operand[0].inv) {
                        oTemp << "~";
                    }
                    oTemp << operand[0].name << " ";
                    if (operand[1].inv) {
                        oTemp << "~";
                    }
                    oTemp << operand[1].name << " \n";

                    oVerilog << "  " << temp_node.name << " [shape=octagon, regular=true];\n";
                    PrintLine(oVerilog, operand[0], temp_node, temp_num);
                    PrintLine(oVerilog, operand[1], temp_node, temp_num);

                    // second part
                    oTemp << "^" << dest.name << " ";
                    oTemp << temp_node.name << " ";
                    if (operand[2].inv) {
                        oTemp << "~";
                    }
                    oTemp << operand[2].name << " \n";

                    oVerilog << "  " << dest.name << " [shape=octagon, regular=true];\n";
                    PrintLine(oVerilog, temp_node, dest, temp_num);
                    PrintLine(oVerilog, operand[2], dest, temp_num);
                }
                continue;
            }

        }
        // Otherwise: majority gate, doesn't need conversion
        else {
            int temp = 0;
            while (rhs[temp] == ' ' || rhs[temp] == '(') {
                temp++;
            }
            while (rhs[temp] != ' ') {
                operand[0].name += rhs[temp];
                temp++;
            }
            CheckOperand(operand[0]);

            while (rhs[temp] == ' ' || rhs[temp] == '&') {
                temp++;
            }
            while (rhs[temp] != ' ') {
                operand[1].name += rhs[temp];
                temp++;
            }
            CheckOperand(operand[1]);

            while (rhs[temp] != '&') {
                temp++;
            }
            temp+=2;
            while (rhs[temp] != ' ') {
                operand[2].name += rhs[temp];
                temp++;
            }
            CheckOperand(operand[2]);

        }
        int temp = lhs.find_first_of("assign");
        temp += 7;
        while (temp < lhs.length() && lhs[temp] != ' ' && lhs[temp] != '=') {
            dest.name += lhs[temp];
            temp++;
        }
        if (dest.name[0] != 'n') dest.leaf = true;

        //////////////

        /*if (dest.leaf) {
            oVerilog << "  " << dest.name << " [shape=none];\n";
        }*/

        for (auto i:operand) {
            PrintLine(oVerilog, i, dest, temp_num);
        }

        oTemp << dest.name << " ";
        for (const auto& i:operand) {
            if (i.name == "0") {
                oTemp << "F ";
            } else if (i.name == "1") {
                oTemp << "T ";
            } else {
                if (i.inv) {
                    oTemp << "~";
                }
                oTemp << i.name << " ";
            }
        }
        oTemp << "\n";
    }

    oVerilog << "}\n";


    oTemp.seekp(0, ios::beg);
    oTemp << x_num << " " << y_num << " " << n_id - n_shift << " " << n_shift;

    cout << filename << ": \t" << n_id - n_shift << "\n";

    iVerilog.close();
    oVerilog.close();
    oTemp.close();
}

void count(const string& input_file = "") {
    int result=0;

    string filename;
    if (input_file.empty()) {
        cout << "Input Verilog filename: ";
        cin >> filename;
    } else {
        filename = input_file;
    }

    ifstream iVerilog;
    iVerilog.open(filename);
    CheckFile((bool) iVerilog, filename);
    string line;
    int minus = 0;

    while (getline(iVerilog, line)) {
        if (line.find("assign") != std::string::npos) {
            result++;
            if (line.find('^') != std::string::npos && line.find("1'b") == std::string::npos) {
                result++;
            }
        } else if (line.find("output") != std::string::npos) {
            int pos = line.find_last_of('y');
            pos++;
            string num_string;
            while (line[pos] != ' ' && line[pos] != ';') {
                num_string += line[pos];
                pos++;
            }
            minus = atoi(num_string.c_str()) + 1;
        }
    }

    result -= minus;

    cout << filename << ": \t" << result << "\n";
}

// Usage: ./converter -v [--xor3] <filename.v>
//        ./converter -d <filename.v>
//        ./converter -c <filename.v>
int main(int argc, char* argv[]) {
    if (argc <= 1) {
        cerr << "Incorrect number of arguments!\n";
        exit(1);
    }
    if (strcmp(argv[1], "-v") == 0) {
        if (argc > 3) {
            cerr << "Incorrect number of arguments!\n";
            exit(1);
        }
        if (argc == 2) {
            v2v();
        } else {
            v2v(argv[2]);
        }
    } else if (strcmp(argv[1], "-d") == 0) {
        if (argc > 4) {
            cerr << "Incorrect number of arguments!\n";
            exit(1);
        }
        if (argc == 2) {
            v2dot(false);
        } else if (argc == 3) {
            if (strcmp(argv[2], "--xor3") == 0) {
                v2dot(true);
            } else {
                v2dot(false, argv[2]);
            }
        } else if (argc == 4) {
            if (strcmp(argv[2], "--xor3") != 0) {
                cerr << "Unrecognized argument: " << argv[2] << "\n";
                exit(3);
            }
            v2dot(true, argv[3]);
        }
    } else if (strcmp(argv[1], "-c") == 0) {
        if (argc == 2) {
            count();
        } else {
            count(argv[2]);
        }
    } else {
        cerr << "Unrecognized command!\n";
        exit(4);
    }

    return 0;
}


PYBIND11_MODULE(converter, m) {
    m.doc() = "pybind11 converter"; // optional module docstring

    m.def("main function", &main, "A function convert verilog to txt");
}

