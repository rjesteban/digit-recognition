
import Jama.*;
import java.io.*;
import com.jmatio.io.MatFileReader;
import com.jmatio.types.*;

class NN {

    private int input_layer_size;
    private int hidden_layer_size;
    private int num_labels;
    private double lambda;
    private int num_iters;
    private Matrix X;
    private Matrix y;
    private int m;
    private double epsilon_init;

    NN(int input_layer_size, int hidden_layer_size, int num_labels, double lambda, int num_iters) {
        this.input_layer_size = input_layer_size;
        this.hidden_layer_size = hidden_layer_size;
        this.lambda = lambda;
        this.num_iters = num_iters;
        this.num_labels = num_labels;
        this.epsilon_init = Math.sqrt(6) / Math.sqrt(input_layer_size + hidden_layer_size);
    }

    public void setnum_labels(int num_labels) {
        this.num_labels = num_labels;
    }

    public int getnum_labels() {
        return num_labels;
    }

    public void setlambda(double lambda) {
        this.lambda = lambda;
    }

    public double getlambda() {
        return lambda;
    }

    public void setnum_iters(int num_iters) {
        this.num_iters = num_iters;
    }

    public int getnum_iters() {
        return num_iters;
    }

    public void setinput_layer_size(int input_layer_size) {
        this.input_layer_size = input_layer_size;
    }

    public int getinput_layer_size() {
        return input_layer_size;
    }

    public void sethidden_layer_size(int hidden_layer_size) {
        this.hidden_layer_size = hidden_layer_size;
    }

    public int gethidden_layer_size() {
        return hidden_layer_size;
    }

    public void setX(Matrix x) {
        this.X = x;
    }

    public Matrix getX() {
        return X;
    }

    public void sety(Matrix y) {
        this.y = y;
    }

    public Matrix gety() {
        return y;
    }

    MatFileReader load(String filename) {
        MatFileReader m = null;
        try {
            m = new MatFileReader(filename);
        } catch (FileNotFoundException e) {
            System.out.println(e.getMessage());
        } catch (IOException ex) {
            System.out.println(ex.getMessage());
        }
        return m;
    }

    double[][] getVariable(MatFileReader reader, String variable) {
        MLDouble j = (MLDouble) reader.getMLArray(variable);
        return j.getArray();
    }

    Matrix gradientDescent(Matrix X, Matrix y, Matrix nn_params, double alpha, int num_iters, int input_layer_size, int hidden_layer_size, int num_labels, double lambda) {
        int m = X.getRowDimension();
        CostFunctionValues c = null;
        int i = 1;
        for (; i <= num_iters; i++) {
            //fill in code here
            c = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
            nn_params = nn_params.minus(c.gradient.times(alpha / m));
            if (i % 10 == 0) {
                System.out.println("Iteration:" + i + " | Cost:" + c.cost);
            }
        }
        System.out.println("Iteration:" + i + " | Cost:" + c.cost);
        return nn_params;
    }

    CostFunctionValues nnCostFunction(Matrix nn_params, int input_layer_size, int hidden_layer_size, int num_labels, Matrix X, Matrix y, double lambda) {

        Matrix theta1 = reshapeTheta1(nn_params, hidden_layer_size, input_layer_size + 1);
        Matrix theta2 = reshapeTheta2(nn_params, num_labels, hidden_layer_size + 1);

        int m = X.getRowDimension();
        double J = 0;//cost

        Matrix theta1Grad = zeros(theta1);
        Matrix theta2Grad = zeros(theta2);

        Matrix subX = X.getMatrix(0, 0, 0, X.getColumnDimension() - 1);

        Matrix a1, z2, a2, z3, h, yi, d3, d2, a, b, t1, t2, t1col1, t2col1, t1col2_to_n, t2col2_to_n, grad;
        CostFunctionValues cfv = new CostFunctionValues();

        for (int i = 0; i < m; i++) {
            //fill in code here
            a1 = this.appendWithOnes(X.getMatrix(0, 0, 0, X.getColumnDimension() - 1), "col").transpose();
            z2 = theta1.times(a1);
            a2 = this.appendWithOnes(this.sigmoid(z2), "row");
            z3 = theta2.times(a2);
            h = sigmoid(z3);

            yi = new Matrix(num_labels, 1);
            yi.set((int) y.get(i, 0) - 1, 0, 1.d);

            d3 = h.minus(yi);
            d2 = theta2.getMatrix(0, theta2.getRowDimension() - 1, 1, theta2.getColumnDimension() - 1);
            d2 = ((d2.transpose()).times(d3)).arrayTimes(sigmoidGradient(z2));

            theta1Grad = theta1Grad.plus(d2.times(a1.transpose()));
            theta2Grad = theta2Grad.plus(d3.times(a2.transpose()));
            J += sum(negateValues(yi).arrayTimes(log(h)).minus(oneminus(yi)).arrayTimes(log(oneminus(h))));

        }
        //fill in code here
        J = J / m;

        t1 = theta1.getMatrix(0, theta1.getRowDimension() - 1, 1, theta1.getColumnDimension() - 1);
        t2 = theta2.getMatrix(0, theta2.getRowDimension() - 1, 1, theta2.getColumnDimension() - 1);

        t1col1 = theta1Grad.getMatrix(0, theta1Grad.getRowDimension() - 1, 0, 0);
        t2col1 = theta2Grad.getMatrix(0, theta2Grad.getRowDimension() - 1, 0, 0);

        t1col2_to_n = theta1Grad.getMatrix(0, theta1Grad.getRowDimension() - 1, 1, theta1Grad.getColumnDimension() - 1);
        t2col2_to_n = theta2Grad.getMatrix(0, theta2Grad.getRowDimension() - 1, 1, theta2Grad.getColumnDimension() - 1);

        t1col2_to_n = t1col2_to_n.plus(t1.times(lambda / m));
        t2col2_to_n = t2col2_to_n.plus(t2.times(lambda / m));

        theta1Grad = joinMatrixByColumns(t1col1, t1col2_to_n);
        theta2Grad = joinMatrixByColumns(t2col1, t2col2_to_n);
        double t1Sum = sum(squared(t1));
        double t2Sum = sum(squared(t2));

        double r = (lambda / (2 * m)) * (t1Sum + t2Sum);
        J = J + r;

        cfv.cost = J;

        grad = unroll(theta1Grad, theta2Grad);
        cfv.gradient = grad;

        return cfv;
    }

    Matrix joinMatrixByColumns(Matrix a, Matrix b) {
        double[][] ar = new double[a.getRowDimension()][a.getColumnDimension() + b.getColumnDimension()];
        for (int i = 0; i < a.getColumnDimension(); i++) {
            for (int j = 0; j < ar[i].length; j++) {
                if (j == 0) {//first column
                    ar[i][j] = a.get(i, 0);
                } else {
                    ar[i][j] = b.get(i, j - 1);
                }
            }
        }
        return new Matrix(ar);
    }

    Matrix zeros(Matrix m) {
        double[][] ar = new double[m.getRowDimension()][m.getColumnDimension()];
        for (int i = 0; i < m.getRowDimension(); i++) {
            for (int j = 0; j < m.getColumnDimension(); j++) {
                ar[i][j] = 0;
            }
        }
        return new Matrix(ar);
    }

    Matrix squared(Matrix m) {
        double[][] ar = new double[m.getRowDimension()][m.getColumnDimension()];
        for (int i = 0; i < m.getRowDimension(); i++) {
            for (int j = 0; j < m.getColumnDimension(); j++) {
                ar[i][j] = m.get(i, j) * m.get(i, j);
            }
        }
        return new Matrix(ar);
    }

    Matrix oneminus(Matrix m) {
        double[][] ar = new double[m.getRowDimension()][m.getColumnDimension()];
        for (int i = 0; i < m.getRowDimension(); i++) {
            for (int j = 0; j < m.getColumnDimension(); j++) {
                ar[i][j] = 1.0 - m.get(i, j);
            }
        }
        return new Matrix(ar);
    }

    Matrix log(Matrix m) {
        double[][] ar = new double[m.getRowDimension()][m.getColumnDimension()];
        for (int i = 0; i < m.getRowDimension(); i++) {
            for (int j = 0; j < m.getColumnDimension(); j++) {
                ar[i][j] = Math.log(m.get(i, j));
            }
        }
        return new Matrix(ar);
    }

    Matrix negateValues(Matrix m) {
        double[][] ar = new double[m.getRowDimension()][m.getColumnDimension()];
        for (int i = 0; i < m.getRowDimension(); i++) {
            for (int j = 0; j < m.getColumnDimension(); j++) {
                ar[i][j] = m.get(i, j) * -1.0;
            }
        }
        return new Matrix(ar);
    }

    public static double sum(Matrix m) {
        double[][] ar = new double[m.getRowDimension()][m.getColumnDimension()];
        double sum = 0;
        for (int i = 0; i < m.getRowDimension(); i++) {
            for (int j = 0; j < m.getColumnDimension(); j++) {
                sum = sum + m.get(i, j);
            }
        }
        return sum;
    }

    Matrix sigmoidGradient(Matrix m) {
        Matrix g = sigmoid(m);
        g = g.arrayTimes(new Matrix(new double[m.getRowDimension()][m.getColumnDimension()]).minus(g));
        return g;
    }

    Matrix sigmoid(Matrix m) {
        int row = m.getRowDimension();
        double[][] ar = new double[row][m.getColumnDimension()];
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < ar[i].length; j++) {
                ar[i][j] = 1 / (1 + Math.exp(-1 * m.get(i, j)));
            }

        }
        return new Matrix(ar);
    }

    Matrix appendWithOnes(Matrix m, String where) {
        double[][] ar = null;
        if (where.equals("col")) {
            ar = new double[m.getRowDimension()][m.getColumnDimension() + 1];
        } else if (where.equals("row")) {
            ar = new double[m.getRowDimension() + 1][m.getColumnDimension()];
        }

        for (int i = 0; i < m.getRowDimension(); i++) {
            for (int j = 0; j < m.getColumnDimension(); j++) {
                if (where.equals("col")) {
                    if (j == 0) {
                        ar[i][j] = 1;
                    } else {
                        ar[i][j] = m.get(i, j);
                    }
                } else if (where.equals("row")) {
                    if (i == 0) {
                        ar[i][j] = 1;
                    } else {
                        ar[i][j] = m.get(i, j);
                    }
                }

            }
        }
        return new Matrix(ar);
    }

    public static void print(double[][] ar) {
        for (int i = 0; i < ar.length; i++) {
            for (int j = 0; j < ar[i].length; j++) {
                System.out.print(ar[i][j] + " ");
            }
            System.out.println();
        }
    }

    public Matrix randInitializeWeights(int input_layer_size, int hidden_layer_size) {
        double[][] a = new double[hidden_layer_size][input_layer_size + 1];

        for (int i = 0; i < hidden_layer_size; i++) {
            for (int j = 0; j < hidden_layer_size + 1; j++) {
                a[i][j] = (Math.random() * (2 * epsilon_init) - epsilon_init);
            }
        }
        return new Matrix(a);
    }

    public Matrix unroll(Matrix initial_Theta1, Matrix initial_Theta2) {

        int rows1 = initial_Theta1.getRowDimension();
        int cols1 = initial_Theta1.getColumnDimension();
        int rows2 = initial_Theta2.getRowDimension();
        int cols2 = initial_Theta2.getColumnDimension();

        double[][] ar = new double[(rows1 * cols1) + (rows2 * cols2)][1];

        for (int i = 0; i < cols1; i++) {
            for (int j = 0; j < rows1; j++) {
                ar[j + (i * rows1)][0] = initial_Theta1.get(j, i);
            }
        }
        int start = cols1 * rows1;
        for (int i = 0; i < cols2; i++) {
            for (int j = 0; j < rows2; j++) {
                ar[start + j + (i * rows2)][0] = initial_Theta2.get(j, i);
            }
        }
        return new Matrix(ar);
    }

    Matrix reshape(Matrix nn_params, int out, int in, int from, int to) {
        int rows = out;
        int cols = in;
        double[][] ar = new double[rows][cols];
        int k = from;
        for (int i = 0; i < cols; i++) {
            for (int j = 0; j < rows; j++) {
                ar[j][i] = nn_params.get(k++, 0);
            }
        }
        return new Matrix(ar);
    }

    Matrix reshapeTheta1(Matrix nn_params, int out, int in) {
        int from = 0;
        int to = out * (in + 1) - 1;
        return reshape(nn_params, out, in, from, to);
    }

    Matrix reshapeTheta2(Matrix nn_params, int out, int in) {
        int to = nn_params.getRowDimension() * nn_params.getColumnDimension();
        int from = to - (in * out);
        return reshape(nn_params, out, in, from, to);
    }

    Matrix predict(Matrix theta1, Matrix theta2, Matrix X) {
        int m = X.getRowDimension();
        int labels = theta2.getRowDimension();
        Matrix p = zeros(X.getMatrix(0, X.getRowDimension() - 1, 0, 0));

        Matrix temp = appendWithOnes(X, "col");

        Matrix h1 = sigmoid(temp.times(theta1.transpose()));
        temp = appendWithOnes(h1, "col");

        Matrix h2 = sigmoid(temp.times(theta2.transpose()));
        p = max(h2);
        return p;
    }

    Matrix max(Matrix m) {
        double[][] ar = new double[m.getRowDimension()][1];
        for (int i = 0; i < ar.length; i++) {
            ar[i][0] = getIndex(m.getMatrix(i, i, 0, m.getColumnDimension() - 1));
        }
        return new Matrix(ar);
    }

    int getIndex(Matrix m) {
        int index = 0;
        double max = m.get(0, 0);
        for (int i = 1; i < m.getColumnDimension(); i++) {
            if (m.get(0, i) > max) {
                max = m.get(0, i);
                index = i;
            }
        }
        return index;
    }

    double accuracy(Matrix pred, Matrix y) {
        int m = pred.getRowDimension();
        double sum = 0.0;
        for (int i = 0; i < m; i++) {
            if (pred.get(i, 0) == y.get(i, 0)) {
                sum++;
            }
        }
        double accuracy = (sum / m) * 100.0;
        return accuracy;
    }
}

class CostFunctionValues {

    double cost;
    Matrix gradient;
}
