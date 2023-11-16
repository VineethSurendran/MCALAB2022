package com.example.spinnerr;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

public class MainActivity extends AppCompatActivity implements AdapterView.OnItemSelectedListener{
    int image[]={R.drawable.fi1, R.drawable.fi2, R.drawable.fi7};
    Spinner sp;
    ImageView i;
    String[] names;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        sp = (Spinner) findViewById(R.id.sp);
        sp.setOnItemSelectedListener(this);
        names = getResources().getStringArray(R.array.Country);
        ArrayAdapter<String> adapter = new ArrayAdapter<>(this,android.R.layout.simple_spinner_item,names);
        sp.setAdapter(adapter);

    }
    @Override
    public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
        i = (ImageView) findViewById(R.id.img);
        i.setImageResource(image[position]);
        TextView myText = (TextView) view;
        Toast.makeText(this,"You selected: "+myText.getText(),Toast.LENGTH_LONG).show();
    }

    @Override
    public void onNothingSelected(AdapterView<?> parent) {

    }
}

