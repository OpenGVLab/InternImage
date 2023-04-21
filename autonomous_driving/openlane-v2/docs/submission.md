# Submission

## Format
The submitted results are required to be stored in a pickle file, which is a dict of identifier and [formatted predictions](../data/README.md#annotations) of a frame:

```
{
    'method':                               <str> -- name of the method
    'authors':                              <list> -- list of str, authors
    'e-mail':                               <str> -- e-mail address
    'institution / company':                <str> -- institution or company
    'country / region':                     <str> -- country or region, checked by iso3166*
    'results': {
        [identifier]: {                     <tuple> -- identifier of the frame, (split, segment_id, timestamp)
            'lane_centerline':              ...
            'traffic_element':              ...
            'topology_lclc':                ...
            'topology_lcte':                ...
        },                       
        ...
    }
}
```
*: For validation, `from iso3166 import countries; countries.get(str)` can be used.

## Steps
1. Create a team on [EvalAI](https://eval.ai/web/challenges/challenge-page/1925).
2. Click the 'Participate' tag, then choose a team for participation.
3. Choose the phase 'Test Phase (CVPR 2023 Autonomous Driving Challenge)' and upload the file formatted as mentioned above.
4. Check if the submitted file is valid, which is indicated by the 'Status' under the tag of 'My Submissions'. A valid submission would provide performance scores.
